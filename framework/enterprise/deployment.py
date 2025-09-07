"""
Enterprise deployment features for multi-GPU inference.
Provides deployment automation, scaling, and enterprise integration capabilities.
"""

import os
import json
import yaml
import subprocess
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

# Optional imports
try:
    import docker
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False
    docker = None

try:
    import kubernetes
    from kubernetes import client, config as k8s_config
    KUBERNETES_AVAILABLE = True
except ImportError:
    KUBERNETES_AVAILABLE = False
    kubernetes = None
    client = None
    k8s_config = None

logger = logging.getLogger(__name__)

class DeploymentType(Enum):
    """Deployment types."""
    STANDALONE = "standalone"
    DOCKER = "docker"
    KUBERNETES = "kubernetes"
    DOCKER_SWARM = "docker_swarm"
    CLOUD = "cloud"

class ScalingMode(Enum):
    """Scaling modes."""
    MANUAL = "manual"
    AUTO_CPU = "auto_cpu"
    AUTO_MEMORY = "auto_memory"
    AUTO_CUSTOM = "auto_custom"

@dataclass
class DeploymentConfig:
    """Deployment configuration."""
    name: str
    type: DeploymentType
    image: str = "torch-inference:latest"
    replicas: int = 1
    resources: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: List[Dict[str, str]] = field(default_factory=list)
    ports: List[int] = field(default_factory=lambda: [8000])
    scaling: Dict[str, Any] = field(default_factory=dict)
    health_check: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CloudConfig:
    """Cloud provider configuration."""
    provider: str  # aws, gcp, azure
    region: str
    instance_type: str
    min_instances: int = 1
    max_instances: int = 10
    auto_scaling: bool = True
    credentials: Dict[str, str] = field(default_factory=dict)

class EnterpriseDeployment:
    """Enterprise deployment manager for multi-GPU inference."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
        self.cloud_configs: Dict[str, CloudConfig] = {}
        
        # Docker client
        self.docker_client = None
        self.k8s_client = None
        
        self._load_configurations()
        self._initialize_clients()
    
    def _load_configurations(self):
        """Load deployment configurations."""
        if self.config_path and os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    if self.config_path.endswith('.yaml') or self.config_path.endswith('.yml'):
                        config_data = yaml.safe_load(f)
                    else:
                        config_data = json.load(f)
                
                # Load deployment configs
                for name, config in config_data.get('deployments', {}).items():
                    self.deployment_configs[name] = DeploymentConfig(
                        name=name,
                        type=DeploymentType(config.get('type', 'standalone')),
                        **{k: v for k, v in config.items() if k != 'type'}
                    )
                
                # Load cloud configs
                for name, config in config_data.get('cloud', {}).items():
                    self.cloud_configs[name] = CloudConfig(**config)
                
                logger.info(f"Loaded {len(self.deployment_configs)} deployment configs")
                
            except Exception as e:
                logger.error(f"Failed to load deployment config: {e}")
    
    def _initialize_clients(self):
        """Initialize deployment clients."""
        # Initialize Docker client
        if DOCKER_AVAILABLE:
            try:
                self.docker_client = docker.from_env()
                logger.info("Docker client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Docker client: {e}")
        else:
            logger.warning("Docker package not available. Install with: pip install docker")
        
        # Initialize Kubernetes client
        if KUBERNETES_AVAILABLE:
            try:
                try:
                    k8s_config.load_incluster_config()  # In-cluster config
                except:
                    k8s_config.load_kube_config()  # Local config
                
                self.k8s_client = client.AppsV1Api()
                logger.info("Kubernetes client initialized")
            except Exception as e:
                logger.warning(f"Failed to initialize Kubernetes client: {e}")
        else:
            logger.warning("Kubernetes package not available. Install with: pip install kubernetes")
    
    def create_docker_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Docker deployment."""
        if not DOCKER_AVAILABLE:
            raise RuntimeError("Docker package not available. Install with: pip install docker")
        
        if not self.docker_client:
            raise RuntimeError("Docker client not available")
        
        try:
            # Prepare container configuration
            container_config = {
                'image': config.image,
                'name': f"{config.name}-{int(time.time())}",
                'environment': config.environment,
                'ports': {f"{port}/tcp": port for port in config.ports},
                'detach': True,
                'restart_policy': {'Name': 'unless-stopped'}
            }
            
            # Add volume mounts
            if config.volumes:
                container_config['volumes'] = {
                    vol['host_path']: {'bind': vol['container_path'], 'mode': vol.get('mode', 'rw')}
                    for vol in config.volumes
                }
            
            # Add resource limits
            if config.resources:
                host_config = {}
                if 'memory' in config.resources:
                    host_config['mem_limit'] = config.resources['memory']
                if 'cpu' in config.resources:
                    host_config['cpu_period'] = 100000
                    host_config['cpu_quota'] = int(config.resources['cpu'] * 100000)
                
                container_config['host_config'] = self.docker_client.api.create_host_config(**host_config)
            
            # Create and start containers
            containers = []
            for i in range(config.replicas):
                container_name = f"{config.name}-{i+1}"
                container = self.docker_client.containers.run(
                    name=container_name,
                    **container_config
                )
                containers.append({
                    'id': container.id,
                    'name': container.name,
                    'status': container.status
                })
            
            logger.info(f"Created Docker deployment '{config.name}' with {config.replicas} replicas")
            
            return {
                'deployment_name': config.name,
                'type': 'docker',
                'containers': containers,
                'status': 'running'
            }
            
        except Exception as e:
            logger.error(f"Failed to create Docker deployment: {e}")
            raise
    
    def create_kubernetes_deployment(self, config: DeploymentConfig) -> Dict[str, Any]:
        """Create Kubernetes deployment."""
        if not KUBERNETES_AVAILABLE:
            raise RuntimeError("Kubernetes package not available. Install with: pip install kubernetes")
        
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        try:
            # Create deployment manifest
            deployment = client.V1Deployment(
                api_version="apps/v1",
                kind="Deployment",
                metadata=client.V1ObjectMeta(name=config.name),
                spec=client.V1DeploymentSpec(
                    replicas=config.replicas,
                    selector=client.V1LabelSelector(
                        match_labels={"app": config.name}
                    ),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(
                            labels={"app": config.name}
                        ),
                        spec=client.V1PodSpec(
                            containers=[
                                client.V1Container(
                                    name=config.name,
                                    image=config.image,
                                    ports=[
                                        client.V1ContainerPort(container_port=port)
                                        for port in config.ports
                                    ],
                                    env=[
                                        client.V1EnvVar(name=k, value=v)
                                        for k, v in config.environment.items()
                                    ],
                                    resources=self._create_k8s_resources(config.resources),
                                    volume_mounts=[
                                        client.V1VolumeMount(
                                            name=f"vol-{i}",
                                            mount_path=vol['container_path']
                                        )
                                        for i, vol in enumerate(config.volumes)
                                    ] if config.volumes else None
                                )
                            ],
                            volumes=[
                                client.V1Volume(
                                    name=f"vol-{i}",
                                    host_path=client.V1HostPathVolumeSource(
                                        path=vol['host_path']
                                    )
                                )
                                for i, vol in enumerate(config.volumes)
                            ] if config.volumes else None
                        )
                    )
                )
            )
            
            # Create deployment
            result = self.k8s_client.create_namespaced_deployment(
                namespace="default",
                body=deployment
            )
            
            # Create service
            service = self._create_k8s_service(config)
            
            logger.info(f"Created Kubernetes deployment '{config.name}'")
            
            return {
                'deployment_name': config.name,
                'type': 'kubernetes',
                'deployment_uid': result.metadata.uid,
                'service_name': service.metadata.name if service else None,
                'status': 'created'
            }
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes deployment: {e}")
            raise
    
    def _create_k8s_resources(self, resources: Dict[str, Any]) -> Optional[Any]:
        """Create Kubernetes resource requirements."""
        if not KUBERNETES_AVAILABLE or not resources:
            return None
        
        requests = {}
        limits = {}
        
        if 'cpu' in resources:
            requests['cpu'] = str(resources['cpu'])
            limits['cpu'] = str(resources.get('cpu_limit', resources['cpu'] * 2))
        
        if 'memory' in resources:
            requests['memory'] = resources['memory']
            limits['memory'] = resources.get('memory_limit', resources['memory'])
        
        if 'nvidia.com/gpu' in resources:
            requests['nvidia.com/gpu'] = str(resources['nvidia.com/gpu'])
            limits['nvidia.com/gpu'] = str(resources['nvidia.com/gpu'])
        
        return client.V1ResourceRequirements(
            requests=requests if requests else None,
            limits=limits if limits else None
        )
    
    def _create_k8s_service(self, config: 'DeploymentConfig') -> Optional[Any]:
        """Create Kubernetes service."""
        if not KUBERNETES_AVAILABLE or not config.ports:
            return None
        
        try:
            service = client.V1Service(
                api_version="v1",
                kind="Service",
                metadata=client.V1ObjectMeta(name=f"{config.name}-service"),
                spec=client.V1ServiceSpec(
                    selector={"app": config.name},
                    ports=[
                        client.V1ServicePort(
                            port=port,
                            target_port=port,
                            protocol="TCP"
                        )
                        for port in config.ports
                    ],
                    type="LoadBalancer"
                )
            )
            
            core_v1 = client.CoreV1Api()
            result = core_v1.create_namespaced_service(
                namespace="default",
                body=service
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create Kubernetes service: {e}")
            return None
    
    def create_deployment(self, config_name: str) -> Dict[str, Any]:
        """Create deployment based on configuration."""
        if config_name not in self.deployment_configs:
            raise ValueError(f"Deployment config '{config_name}' not found")
        
        config = self.deployment_configs[config_name]
        
        if config.type == DeploymentType.DOCKER:
            return self.create_docker_deployment(config)
        elif config.type == DeploymentType.KUBERNETES:
            return self.create_kubernetes_deployment(config)
        else:
            raise ValueError(f"Deployment type '{config.type}' not supported")
    
    def scale_deployment(self, deployment_name: str, replicas: int, 
                        deployment_type: DeploymentType = DeploymentType.KUBERNETES) -> Dict[str, Any]:
        """Scale existing deployment."""
        try:
            if deployment_type == DeploymentType.KUBERNETES:
                # Scale Kubernetes deployment
                self.k8s_client.patch_namespaced_deployment_scale(
                    name=deployment_name,
                    namespace="default",
                    body=client.V1Scale(
                        spec=client.V1ScaleSpec(replicas=replicas)
                    )
                )
                
                return {
                    'deployment_name': deployment_name,
                    'new_replicas': replicas,
                    'status': 'scaling'
                }
            
            elif deployment_type == DeploymentType.DOCKER:
                # For Docker, we'd need to implement custom scaling logic
                # This would involve creating/removing containers
                containers = self.docker_client.containers.list(
                    filters={'name': deployment_name}
                )
                
                current_replicas = len(containers)
                
                if replicas > current_replicas:
                    # Scale up - create more containers
                    # Implementation would depend on original container config
                    pass
                elif replicas < current_replicas:
                    # Scale down - remove containers
                    for i in range(current_replicas - replicas):
                        containers[i].stop()
                        containers[i].remove()
                
                return {
                    'deployment_name': deployment_name,
                    'old_replicas': current_replicas,
                    'new_replicas': replicas,
                    'status': 'scaled'
                }
            
        except Exception as e:
            logger.error(f"Failed to scale deployment: {e}")
            raise
    
    def get_deployment_status(self, deployment_name: str, 
                            deployment_type: DeploymentType = DeploymentType.KUBERNETES) -> Dict[str, Any]:
        """Get deployment status."""
        try:
            if deployment_type == DeploymentType.KUBERNETES:
                deployment = self.k8s_client.read_namespaced_deployment_status(
                    name=deployment_name,
                    namespace="default"
                )
                
                return {
                    'name': deployment.metadata.name,
                    'replicas': deployment.spec.replicas,
                    'ready_replicas': deployment.status.ready_replicas or 0,
                    'available_replicas': deployment.status.available_replicas or 0,
                    'updated_replicas': deployment.status.updated_replicas or 0,
                    'conditions': [
                        {
                            'type': condition.type,
                            'status': condition.status,
                            'reason': condition.reason,
                            'message': condition.message
                        }
                        for condition in (deployment.status.conditions or [])
                    ]
                }
            
            elif deployment_type == DeploymentType.DOCKER:
                containers = self.docker_client.containers.list(
                    filters={'name': deployment_name}
                )
                
                return {
                    'name': deployment_name,
                    'containers': [
                        {
                            'id': container.id[:12],
                            'name': container.name,
                            'status': container.status,
                            'image': container.image.tags[0] if container.image.tags else 'unknown'
                        }
                        for container in containers
                    ],
                    'total_containers': len(containers)
                }
            
        except Exception as e:
            logger.error(f"Failed to get deployment status: {e}")
            raise
    
    def delete_deployment(self, deployment_name: str, 
                         deployment_type: DeploymentType = DeploymentType.KUBERNETES) -> Dict[str, Any]:
        """Delete deployment."""
        try:
            if deployment_type == DeploymentType.KUBERNETES:
                # Delete deployment
                self.k8s_client.delete_namespaced_deployment(
                    name=deployment_name,
                    namespace="default"
                )
                
                # Delete service
                try:
                    core_v1 = client.CoreV1Api()
                    core_v1.delete_namespaced_service(
                        name=f"{deployment_name}-service",
                        namespace="default"
                    )
                except:
                    pass  # Service might not exist
                
                return {
                    'deployment_name': deployment_name,
                    'status': 'deleted'
                }
            
            elif deployment_type == DeploymentType.DOCKER:
                containers = self.docker_client.containers.list(
                    filters={'name': deployment_name}
                )
                
                for container in containers:
                    container.stop()
                    container.remove()
                
                return {
                    'deployment_name': deployment_name,
                    'containers_removed': len(containers),
                    'status': 'deleted'
                }
            
        except Exception as e:
            logger.error(f"Failed to delete deployment: {e}")
            raise
    
    def generate_deployment_manifest(self, config_name: str, output_path: str = None) -> str:
        """Generate deployment manifest."""
        if config_name not in self.deployment_configs:
            raise ValueError(f"Deployment config '{config_name}' not found")
        
        config = self.deployment_configs[config_name]
        
        if config.type == DeploymentType.KUBERNETES:
            manifest = self._generate_k8s_manifest(config)
        elif config.type == DeploymentType.DOCKER:
            manifest = self._generate_docker_compose(config)
        else:
            raise ValueError(f"Manifest generation not supported for {config.type}")
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(manifest)
            logger.info(f"Deployment manifest written to {output_path}")
        
        return manifest
    
    def _generate_k8s_manifest(self, config: DeploymentConfig) -> str:
        """Generate Kubernetes manifest."""
        deployment_manifest = {
            'apiVersion': 'apps/v1',
            'kind': 'Deployment',
            'metadata': {'name': config.name},
            'spec': {
                'replicas': config.replicas,
                'selector': {'matchLabels': {'app': config.name}},
                'template': {
                    'metadata': {'labels': {'app': config.name}},
                    'spec': {
                        'containers': [{
                            'name': config.name,
                            'image': config.image,
                            'ports': [{'containerPort': port} for port in config.ports],
                            'env': [{'name': k, 'value': v} for k, v in config.environment.items()]
                        }]
                    }
                }
            }
        }
        
        service_manifest = {
            'apiVersion': 'v1',
            'kind': 'Service',
            'metadata': {'name': f"{config.name}-service"},
            'spec': {
                'selector': {'app': config.name},
                'ports': [{'port': port, 'targetPort': port} for port in config.ports],
                'type': 'LoadBalancer'
            }
        }
        
        return yaml.dump_all([deployment_manifest, service_manifest], default_flow_style=False)
    
    def _generate_docker_compose(self, config: DeploymentConfig) -> str:
        """Generate Docker Compose manifest."""
        compose = {
            'version': '3.8',
            'services': {
                config.name: {
                    'image': config.image,
                    'ports': [f"{port}:{port}" for port in config.ports],
                    'environment': config.environment,
                    'deploy': {
                        'replicas': config.replicas,
                        'restart_policy': {'condition': 'on-failure'}
                    }
                }
            }
        }
        
        if config.volumes:
            compose['services'][config.name]['volumes'] = [
                f"{vol['host_path']}:{vol['container_path']}"
                for vol in config.volumes
            ]
        
        return yaml.dump(compose, default_flow_style=False)
    
    def setup_auto_scaling(self, deployment_name: str, min_replicas: int = 1, 
                          max_replicas: int = 10, cpu_threshold: int = 70) -> Dict[str, Any]:
        """Setup auto-scaling for Kubernetes deployment."""
        if not self.k8s_client:
            raise RuntimeError("Kubernetes client not available")
        
        try:
            autoscaling_v1 = client.AutoscalingV1Api()
            
            hpa = client.V1HorizontalPodAutoscaler(
                api_version="autoscaling/v1",
                kind="HorizontalPodAutoscaler",
                metadata=client.V1ObjectMeta(name=f"{deployment_name}-hpa"),
                spec=client.V1HorizontalPodAutoscalerSpec(
                    scale_target_ref=client.V1CrossVersionObjectReference(
                        api_version="apps/v1",
                        kind="Deployment",
                        name=deployment_name
                    ),
                    min_replicas=min_replicas,
                    max_replicas=max_replicas,
                    target_cpu_utilization_percentage=cpu_threshold
                )
            )
            
            result = autoscaling_v1.create_namespaced_horizontal_pod_autoscaler(
                namespace="default",
                body=hpa
            )
            
            return {
                'deployment_name': deployment_name,
                'hpa_name': result.metadata.name,
                'min_replicas': min_replicas,
                'max_replicas': max_replicas,
                'cpu_threshold': cpu_threshold,
                'status': 'created'
            }
            
        except Exception as e:
            logger.error(f"Failed to setup auto-scaling: {e}")
            raise
    
    def get_deployment_logs(self, deployment_name: str, 
                           deployment_type: DeploymentType = DeploymentType.KUBERNETES,
                           lines: int = 100) -> List[str]:
        """Get deployment logs."""
        try:
            if deployment_type == DeploymentType.KUBERNETES:
                core_v1 = client.CoreV1Api()
                
                # Get pods for deployment
                pods = core_v1.list_namespaced_pod(
                    namespace="default",
                    label_selector=f"app={deployment_name}"
                )
                
                logs = []
                for pod in pods.items:
                    pod_logs = core_v1.read_namespaced_pod_log(
                        name=pod.metadata.name,
                        namespace="default",
                        tail_lines=lines
                    )
                    logs.extend(pod_logs.split('\n'))
                
                return logs
            
            elif deployment_type == DeploymentType.DOCKER:
                containers = self.docker_client.containers.list(
                    filters={'name': deployment_name}
                )
                
                logs = []
                for container in containers:
                    container_logs = container.logs(tail=lines).decode('utf-8')
                    logs.extend(container_logs.split('\n'))
                
                return logs
            
        except Exception as e:
            logger.error(f"Failed to get deployment logs: {e}")
            raise
    
    def list_deployments(self, deployment_type: DeploymentType = DeploymentType.KUBERNETES) -> List[Dict[str, Any]]:
        """List all deployments."""
        try:
            if deployment_type == DeploymentType.KUBERNETES:
                deployments = self.k8s_client.list_namespaced_deployment(namespace="default")
                
                return [
                    {
                        'name': deployment.metadata.name,
                        'replicas': deployment.spec.replicas,
                        'ready_replicas': deployment.status.ready_replicas or 0,
                        'created': deployment.metadata.creation_timestamp.isoformat(),
                        'image': deployment.spec.template.spec.containers[0].image
                    }
                    for deployment in deployments.items
                ]
            
            elif deployment_type == DeploymentType.DOCKER:
                containers = self.docker_client.containers.list()
                
                # Group containers by deployment name (extracted from container name)
                deployments = {}
                for container in containers:
                    # Assume container names follow pattern: deployment-name-replica-number
                    parts = container.name.split('-')
                    if len(parts) >= 2:
                        deployment_name = '-'.join(parts[:-1])
                        if deployment_name not in deployments:
                            deployments[deployment_name] = []
                        deployments[deployment_name].append(container)
                
                return [
                    {
                        'name': name,
                        'replicas': len(containers),
                        'containers': [
                            {
                                'id': c.id[:12],
                                'status': c.status,
                                'image': c.image.tags[0] if c.image.tags else 'unknown'
                            }
                            for c in containers
                        ]
                    }
                    for name, containers in deployments.items()
                ]
            
        except Exception as e:
            logger.error(f"Failed to list deployments: {e}")
            raise
    
    def cleanup(self):
        """Clean up deployment manager resources."""
        if self.docker_client:
            self.docker_client.close()
        
        logger.info("Enterprise deployment manager cleanup completed")
