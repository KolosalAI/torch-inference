"""
Advanced Alerting System

Provides:
- Multi-channel alerting (email, Slack, PagerDuty, webhooks)
- Alert severity levels and escalation
- Alert aggregation and deduplication
- Notification rate limiting
- Alert routing based on conditions
"""

import asyncio
import logging
import smtplib
import json
import time
from typing import Any, Dict, List, Optional, Set, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from abc import ABC, abstractmethod
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import hashlib
import threading
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    
    def __eq__(self, other):
        """Allow comparison with string values."""
        if isinstance(other, str):
            return self.value == other
        return super().__eq__(other)
    
    def __hash__(self):
        """Support hashing."""
        return super().__hash__()
    
    def __str__(self):
        """Return the value when converted to string."""
        return self.value


class AlertStatus(Enum):
    """Alert status."""
    ACTIVE = "active"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SUPPRESSED = "suppressed"


@dataclass
class Alert:
    """Alert definition."""
    id: str
    title: str
    message: str  # Changed from 'description' to match test expectations
    severity: AlertSeverity
    source: str = "unknown"  # Make source optional with default
    metadata: Dict[str, Any] = field(default_factory=dict)  # Add metadata field
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    status: AlertStatus = AlertStatus.ACTIVE
    fingerprint: Optional[str] = None
    resolved_at: Optional[datetime] = None
    resolution_message: str = ""  # Add resolution message field
    
    def __post_init__(self):
        """Generate fingerprint for deduplication."""
        if not self.fingerprint:
            # Create fingerprint from title, source, and labels
            fingerprint_data = f"{self.title}:{self.source}:{json.dumps(sorted(self.labels.items()))}"
            self.fingerprint = hashlib.md5(fingerprint_data.encode()).hexdigest()
    
    @property
    def timestamp(self) -> datetime:
        """Get timestamp (alias for created_at)."""
        return self.created_at
    
    @property
    def resolved(self) -> bool:
        """Check if alert is resolved."""
        return self.status == AlertStatus.RESOLVED
    
    @property
    def description(self) -> str:
        """Get description (alias for message)."""
        return self.message
    
    @property
    def age(self) -> timedelta:
        """Get age of the alert."""
        return datetime.utcnow() - self.created_at
    
    def resolve(self, resolution_note: str = ""):
        """Resolve the alert."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()
        self.resolution_message = resolution_note
        if resolution_note:
            self.annotations['resolution'] = resolution_note
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "id": self.id,
            "title": self.title,
            "message": self.message,
            "description": self.description,
            "severity": self.severity.value.upper(),  # Convert to uppercase
            "source": self.source,
            "metadata": self.metadata,
            "labels": self.labels,
            "annotations": self.annotations,
            "timestamp": self.timestamp.isoformat(),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "status": self.status.value,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_message": self.resolution_message,
            "fingerprint": self.fingerprint
        }


@dataclass
class AlertRule:
    """Alert rule definition."""
    name: str
    condition: Callable[[Dict[str, Any]], bool]
    severity: AlertSeverity
    title: str  # Changed from title_template to match test expectations
    message_template: str  # Changed from description_template to match test expectations
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)
    enabled: bool = True
    cooldown_period: float = 300.0  # Changed from cooldown_seconds to match test expectations
    last_triggered: Optional[datetime] = None
    
    def evaluate(self, metrics: Dict[str, Any]) -> tuple[bool, Optional[Alert]]:
        """Evaluate rule against metrics and return (should_trigger, alert)."""
        if not self.enabled:
            return False, None
        
        # Check cooldown period
        now = datetime.utcnow()
        if (self.last_triggered and 
            (now - self.last_triggered).total_seconds() < self.cooldown_period):
            return False, None
        
        try:
            # Evaluate condition
            if not self.condition(metrics):
                return False, None
            
            # Create alert
            alert_id = f"{self.name}_{int(time.time())}"
            
            # Format message template
            try:
                message = self.message_template.format(**metrics)
            except (KeyError, ValueError) as e:
                logger.warning(f"Error formatting message template for rule {self.name}: {e}")
                message = f"Alert triggered by rule {self.name}"
            
            alert = Alert(
                id=alert_id,
                title=self.title,
                message=message,
                severity=self.severity,
                source=f"rule:{self.name}",
                labels=self.labels.copy(),
                annotations=self.annotations.copy()
            )
            
            # Update last triggered time
            self.last_triggered = now
            
            return True, alert
            
        except Exception as e:
            logger.error(f"Error evaluating rule {self.name}: {e}")
            return False, None


@dataclass
class NotificationChannel:
    """Notification channel configuration."""
    name: str
    type: str
    config: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


@dataclass
class AlertRoute:
    """Alert routing configuration."""
    name: str
    matchers: Dict[str, Union[str, List[str]]]  # Label matchers
    channels: List[str]  # Channel names
    severity_filter: Optional[List[AlertSeverity]] = None
    enabled: bool = True


class AlertNotifier(ABC):
    """Abstract base class for alert notifiers."""
    
    @abstractmethod
    async def send_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert notification. Returns True if successful."""
        pass
    
    @abstractmethod
    async def send_resolved(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send alert resolved notification. Returns True if successful."""
        pass


class EmailNotifier(AlertNotifier):
    """Email alert notifier."""
    
    async def send_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email alert."""
        try:
            config = channel.config
            smtp_host = config.get('smtp_host', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            smtp_user = config.get('smtp_user')
            smtp_password = config.get('smtp_password')
            use_tls = config.get('use_tls', True)
            
            from_email = config.get('from_email', smtp_user)
            to_emails = config.get('to_emails', [])
            
            if not to_emails:
                logger.warning("No recipient emails configured")
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[{alert.severity.value.upper()}] {alert.title}"
            
            # Email body
            body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Source: {alert.source}
Time: {alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC')}

Description:
{alert.description}

Labels:
{json.dumps(alert.labels, indent=2)}

Annotations:
{json.dumps(alert.annotations, indent=2)}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")
            return False
    
    async def send_resolved(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send email resolved notification."""
        try:
            config = channel.config
            smtp_host = config.get('smtp_host', 'localhost')
            smtp_port = config.get('smtp_port', 587)
            smtp_user = config.get('smtp_user')
            smtp_password = config.get('smtp_password')
            use_tls = config.get('use_tls', True)
            
            from_email = config.get('from_email', smtp_user)
            to_emails = config.get('to_emails', [])
            
            if not to_emails:
                return False
            
            # Create email
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = ', '.join(to_emails)
            msg['Subject'] = f"[RESOLVED] {alert.title}"
            
            # Email body
            body = f"""
Alert RESOLVED: {alert.title}
Source: {alert.source}
Resolved at: {alert.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC')}
Duration: {alert.updated_at - alert.created_at}

Original Description:
{alert.description}
"""
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Send email
            with smtplib.SMTP(smtp_host, smtp_port) as server:
                if use_tls:
                    server.starttls()
                if smtp_user and smtp_password:
                    server.login(smtp_user, smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email resolved notification sent for {alert.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email resolved notification: {e}")
            return False


class SlackNotifier(AlertNotifier):
    """Slack alert notifier."""
    
    async def send_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack alert."""
        try:
            import aiohttp
            
            webhook_url = channel.config.get('webhook_url')
            if not webhook_url:
                logger.warning("No Slack webhook URL configured")
                return False
            
            # Color based on severity
            color_map = {
                AlertSeverity.CRITICAL: "danger",
                AlertSeverity.WARNING: "warning", 
                AlertSeverity.INFO: "good"
            }
            
            # Create Slack message
            message = {
                "text": f"Alert: {alert.title}",
                "attachments": [
                    {
                        "color": color_map.get(alert.severity, "warning"),
                        "fields": [
                            {"title": "Severity", "value": alert.severity.value.upper(), "short": True},
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Time", "value": alert.created_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": False},
                            {"title": "Description", "value": alert.description, "short": False}
                        ]
                    }
                ]
            }
            
            # Add labels and annotations if present
            if alert.labels:
                message["attachments"][0]["fields"].append({
                    "title": "Labels",
                    "value": "\n".join([f"• {k}: {v}" for k, v in alert.labels.items()]),
                    "short": False
                })
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    if response.status == 200:
                        logger.info(f"Slack alert sent for {alert.id}")
                        return True
                    else:
                        logger.error(f"Failed to send Slack alert: HTTP {response.status}")
                        return False
                        
        except ImportError:
            logger.error("aiohttp required for Slack notifications")
            return False
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
            return False
    
    async def send_resolved(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send Slack resolved notification."""
        try:
            import aiohttp
            
            webhook_url = channel.config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Create resolved message
            duration = alert.updated_at - alert.created_at
            message = {
                "text": f"✅ Alert Resolved: {alert.title}",
                "attachments": [
                    {
                        "color": "good",
                        "fields": [
                            {"title": "Source", "value": alert.source, "short": True},
                            {"title": "Duration", "value": str(duration), "short": True},
                            {"title": "Resolved at", "value": alert.updated_at.strftime('%Y-%m-%d %H:%M:%S UTC'), "short": False}
                        ]
                    }
                ]
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=message) as response:
                    return response.status == 200
                        
        except Exception as e:
            logger.error(f"Failed to send Slack resolved notification: {e}")
            return False


class WebhookNotifier(AlertNotifier):
    """Generic webhook notifier."""
    
    async def send_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook alert."""
        try:
            import aiohttp
            
            webhook_url = channel.config.get('webhook_url')
            if not webhook_url:
                logger.warning("No webhook URL configured")
                return False
            
            # Create webhook payload
            payload = {
                "alert": {
                    "id": alert.id,
                    "title": alert.title,
                    "description": alert.description,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "source": alert.source,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "created_at": alert.created_at.isoformat(),
                    "fingerprint": alert.fingerprint
                },
                "type": "alert"
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "TorchInference-AlertManager/1.0"
            }
            
            # Add custom headers if configured
            custom_headers = channel.config.get('headers', {})
            headers.update(custom_headers)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    if response.status < 400:
                        logger.info(f"Webhook alert sent for {alert.id}")
                        return True
                    else:
                        logger.error(f"Failed to send webhook alert: HTTP {response.status}")
                        return False
                        
        except ImportError:
            logger.error("aiohttp required for webhook notifications")
            return False
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
            return False
    
    async def send_resolved(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send webhook resolved notification."""
        try:
            import aiohttp
            
            webhook_url = channel.config.get('webhook_url')
            if not webhook_url:
                return False
            
            # Create resolved payload
            payload = {
                "alert": {
                    "id": alert.id,
                    "title": alert.title,
                    "severity": alert.severity.value,
                    "status": alert.status.value,
                    "source": alert.source,
                    "labels": alert.labels,
                    "created_at": alert.created_at.isoformat(),
                    "updated_at": alert.updated_at.isoformat(),
                    "fingerprint": alert.fingerprint
                },
                "type": "resolved"
            }
            
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "TorchInference-AlertManager/1.0"
            }
            
            custom_headers = channel.config.get('headers', {})
            headers.update(custom_headers)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(webhook_url, json=payload, headers=headers) as response:
                    return response.status < 400
                        
        except Exception as e:
            logger.error(f"Failed to send webhook resolved notification: {e}")
            return False


class PagerDutyNotifier(AlertNotifier):
    """PagerDuty alert notifier."""
    
    async def send_alert(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send PagerDuty alert."""
        try:
            import aiohttp
            
            integration_key = channel.config.get('integration_key')
            if not integration_key:
                logger.warning("No PagerDuty integration key configured")
                return False
            
            # Map severity to PagerDuty severity
            severity_map = {
                AlertSeverity.CRITICAL: "critical",
                AlertSeverity.WARNING: "warning",
                AlertSeverity.INFO: "info"
            }
            
            # Create PagerDuty event
            event = {
                "routing_key": integration_key,
                "event_action": "trigger",
                "dedup_key": alert.fingerprint,
                "payload": {
                    "summary": alert.title,
                    "source": alert.source,
                    "severity": severity_map.get(alert.severity, "warning"),
                    "timestamp": alert.created_at.isoformat(),
                    "custom_details": {
                        "description": alert.description,
                        "labels": alert.labels,
                        "annotations": alert.annotations,
                        "alert_id": alert.id
                    }
                }
            }
            
            url = "https://events.pagerduty.com/v2/enqueue"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=event) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty alert sent for {alert.id}")
                        return True
                    else:
                        logger.error(f"Failed to send PagerDuty alert: HTTP {response.status}")
                        return False
                        
        except ImportError:
            logger.error("aiohttp required for PagerDuty notifications")
            return False
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
            return False
    
    async def send_resolved(self, alert: Alert, channel: NotificationChannel) -> bool:
        """Send PagerDuty resolved notification."""
        try:
            import aiohttp
            
            integration_key = channel.config.get('integration_key')
            if not integration_key:
                return False
            
            # Create PagerDuty resolve event
            event = {
                "routing_key": integration_key,
                "event_action": "resolve",
                "dedup_key": alert.fingerprint
            }
            
            url = "https://events.pagerduty.com/v2/enqueue"
            
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=event) as response:
                    return response.status == 202
                        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty resolved notification: {e}")
            return False


class AlertAggregator:
    """Alert aggregation and deduplication."""
    
    def __init__(self, window_seconds: float = 60.0, max_alerts: int = 10):
        self.window_seconds = window_seconds
        self.max_alerts = max_alerts
        self._aggregated_alerts: Dict[str, List[Alert]] = defaultdict(list)
        self._last_sent: Dict[str, datetime] = {}
        self._lock = threading.RLock()
    
    def should_aggregate(self, alert: Alert) -> bool:
        """Check if alert should be aggregated."""
        with self._lock:
            fingerprint = alert.fingerprint
            now = datetime.utcnow()
            
            # Clean old alerts
            self._clean_old_alerts(fingerprint, now)
            
            # Check if we have similar alerts recently
            recent_alerts = self._aggregated_alerts[fingerprint]
            
            if len(recent_alerts) >= self.max_alerts:
                return True
            
            # Check time-based aggregation
            last_sent = self._last_sent.get(fingerprint)
            if last_sent and (now - last_sent).total_seconds() < self.window_seconds:
                return True
            
            return False
    
    def add_alert(self, alert: Alert):
        """Add alert to aggregation buffer."""
        with self._lock:
            fingerprint = alert.fingerprint
            self._aggregated_alerts[fingerprint].append(alert)
    
    def get_aggregated_alert(self, fingerprint: str) -> Optional[Alert]:
        """Get aggregated alert for a fingerprint."""
        with self._lock:
            alerts = self._aggregated_alerts.get(fingerprint, [])
            if not alerts:
                return None
            
            # Create aggregated alert
            first_alert = alerts[0]
            last_alert = alerts[-1]
            
            aggregated = Alert(
                id=f"aggregated_{fingerprint}_{int(time.time())}",
                title=f"{first_alert.title} (x{len(alerts)})",
                description=f"Aggregated {len(alerts)} similar alerts.\n\nLatest: {last_alert.description}",
                severity=max(alert.severity for alert in alerts),  # Highest severity
                source=first_alert.source,
                labels=first_alert.labels,
                annotations={
                    **first_alert.annotations,
                    "aggregated_count": str(len(alerts)),
                    "first_occurrence": first_alert.created_at.isoformat(),
                    "last_occurrence": last_alert.created_at.isoformat()
                },
                created_at=first_alert.created_at,
                fingerprint=fingerprint
            )
            
            # Clear aggregation buffer
            self._aggregated_alerts[fingerprint] = []
            self._last_sent[fingerprint] = datetime.utcnow()
            
            return aggregated
    
    def _clean_old_alerts(self, fingerprint: str, now: datetime):
        """Clean old alerts from aggregation buffer."""
        alerts = self._aggregated_alerts[fingerprint]
        cutoff = now - timedelta(seconds=self.window_seconds)
        
        # Keep only recent alerts
        self._aggregated_alerts[fingerprint] = [
            alert for alert in alerts
            if alert.created_at > cutoff
        ]


class RateLimiter:
    """Rate limiter for notifications."""
    
    def __init__(self, max_notifications: int = 10, window_seconds: float = 3600.0):
        self.max_notifications = max_notifications
        self.window_seconds = window_seconds
        self._notification_times: Dict[str, deque] = defaultdict(deque)
        self._lock = threading.RLock()
    
    def can_send(self, channel_name: str) -> bool:
        """Check if we can send a notification to this channel."""
        with self._lock:
            now = time.time()
            times = self._notification_times[channel_name]
            
            # Remove old timestamps
            cutoff = now - self.window_seconds
            while times and times[0] < cutoff:
                times.popleft()
            
            return len(times) < self.max_notifications
    
    def record_notification(self, channel_name: str):
        """Record a notification sent to a channel."""
        with self._lock:
            self._notification_times[channel_name].append(time.time())


# Compatibility aliases for the old channel classes
class EmailChannel(EmailNotifier):
    """Email notification channel - compatibility alias."""
    
    def __init__(self, smtp_server: str, smtp_port: int = 587, username: str = None, 
                 password: str = None, from_address: str = None, to_addresses: List[str] = None):
        self.config = {
            'smtp_host': smtp_server,
            'smtp_port': smtp_port,
            'smtp_user': username,
            'smtp_password': password,
            'from_email': from_address,
            'to_emails': to_addresses or [],
            'use_tls': True
        }
        self.channel = NotificationChannel(
            name="email",
            type="email",
            config=self.config
        )
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification using the channel config."""
        return await self.send_alert(alert, self.channel)
    
    def _format_message(self, alert: Alert) -> tuple[str, str]:
        """Format email message for alert."""
        subject = f"[{alert.severity.value.upper()}] {alert.title}"
        
        body = f"""
Alert: {alert.title}
Severity: {alert.severity.value.upper()}
Description: {alert.description}
Time: {alert.created_at}

Metadata:
"""
        
        for key, value in alert.metadata.items():
            body += f"  {key}: {value}\n"
        
        return subject, body


class SlackChannel(SlackNotifier):
    """Slack notification channel - compatibility alias."""
    
    def __init__(self, webhook_url: str, channel: str = None, username: str = None):
        self.config = {
            'webhook_url': webhook_url,
            'channel': channel,
            'username': username
        }
        self.channel = NotificationChannel(
            name="slack",
            type="slack", 
            config=self.config
        )
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification using the channel config."""
        return await self.send_alert(alert, self.channel)
    
    def _format_message(self, alert: Alert) -> dict:
        """Format Slack message for alert."""
        severity_emoji = {
            'critical': '🔴',
            'error': '🟠',
            'warning': '🟡',
            'info': '🔵'
        }
        
        emoji = severity_emoji.get(alert.severity.value, '⚪')
        
        text = f"{emoji} *{alert.severity.value.upper()}*: {alert.title}\n"
        text += f"{alert.description}\n"
        text += f"Time: {alert.created_at}\n"
        
        if alert.metadata:
            text += "\n*Metadata:*\n"
            for key, value in alert.metadata.items():
                text += f"• {key}: {value}\n"
        
        return {
            "channel": self.config.get('channel', '#alerts'),
            "username": self.config.get('username', 'AlertBot'),
            "text": text
        }


class PagerDutyChannel(PagerDutyNotifier):
    """PagerDuty notification channel - compatibility alias."""
    
    def __init__(self, integration_key: str, service_name: str = None):
        self.config = {
            'integration_key': integration_key,
            'service_name': service_name
        }
        self.channel = NotificationChannel(
            name="pagerduty",
            type="pagerduty",
            config=self.config
        )
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification using the channel config."""
        return await self.send_alert(alert, self.channel)
    
    def _map_severity(self, severity: AlertSeverity) -> str:
        """Map alert severity to PagerDuty severity."""
        mapping = {
            AlertSeverity.CRITICAL: "critical",
            AlertSeverity.ERROR: "error",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.INFO: "info"
        }
        return mapping.get(severity, "warning")


class WebhookChannel(WebhookNotifier):
    """Webhook notification channel - compatibility alias."""
    
    def __init__(self, webhook_url: str, headers: Dict[str, str] = None, payload_formatter: Callable = None):
        self.config = {
            'webhook_url': webhook_url,
            'headers': headers or {}
        }
        self.channel = NotificationChannel(
            name="webhook",
            type="webhook",
            config=self.config
        )
        self.payload_formatter = payload_formatter
    
    async def send_notification(self, alert: Alert) -> bool:
        """Send notification using the channel config."""
        import aiohttp
        
        try:
            # Format message using our method
            payload = self._format_message(alert)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config['webhook_url'],
                    json=payload,
                    headers=self.config.get('headers', {})
                ) as response:
                    return 200 <= response.status < 300
        except Exception as e:
            logger.error(f"Failed to send webhook notification: {e}")
            return False
    
    def _format_message(self, alert: Alert) -> dict:
        """Format webhook message for alert."""
        if self.payload_formatter:
            return self.payload_formatter(alert)
        
        # Default payload format
        return {
            "alert_id": alert.id,
            "title": alert.title,
            "description": alert.description,
            "severity": alert.severity.value,
            "timestamp": alert.created_at.isoformat(),
            "metadata": alert.metadata,
            "status": alert.status.value
        }


class AlertManager:
    """
    Advanced alerting system with multi-channel notifications.
    
    Features:
    - Multiple notification channels (email, Slack, PagerDuty, webhooks)
    - Alert routing based on labels and severity
    - Alert aggregation and deduplication
    - Rate limiting
    - Alert lifecycle management
    """
    
    def __init__(self):
        # Alert storage
        self._active_alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._rules: Dict[str, AlertRule] = {}  # Changed from _alert_rules to match tests
        
        # Notification system
        self._channels: Dict[str, NotificationChannel] = {}
        self._routes: List[AlertRoute] = []
        self._notifiers = {
            'email': EmailNotifier(),
            'slack': SlackNotifier(),
            'webhook': WebhookNotifier(),
            'pagerduty': PagerDutyNotifier()
        }
        
        # Rate limiting and aggregation
        self.rate_limiter = RateLimiter()
        self.aggregator = AlertAggregator()
        
        # Background tasks
        self._evaluation_task: Optional[asyncio.Task] = None
        self._aggregation_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        logger.info("Alert manager initialized")
    
    def add_channel(self, name: str, channel: NotificationChannel):
        """Add a notification channel."""
        with self._lock:
            self._channels[name] = channel
        
        # Get channel type safely for logging
        channel_type = getattr(channel, 'type', 'unknown')
        logger.info(f"Added notification channel: {name} ({channel_type})")
    
    def remove_channel(self, name: str):
        """Remove a notification channel."""
        with self._lock:
            if name in self._channels:
                del self._channels[name]
                logger.info(f"Removed notification channel: {name}")
            else:
                raise ValueError(f"Channel '{name}' not found")
    
    def add_route(self, route: AlertRoute):
        """Add an alert route."""
        with self._lock:
            self._routes.append(route)
        
        logger.info(f"Added alert route: {route.name}")
    
    def add_rule(self, rule: AlertRule):
        """Add an alert rule."""
        with self._lock:
            if rule.name in self._rules:
                raise ValueError(f"Rule '{rule.name}' already exists")
            self._rules[rule.name] = rule
        
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, name: str):
        """Remove an alert rule."""
        with self._lock:
            if name in self._rules:
                del self._rules[name]
                logger.info(f"Removed alert rule: {name}")
            else:
                raise ValueError(f"Rule '{name}' not found")
    
    async def fire_alert(self, alert: Alert) -> str:
        """Fire an alert."""
        with self._lock:
            # Check if alert already exists (deduplication)
            existing_alert = self._active_alerts.get(alert.fingerprint)
            if existing_alert:
                # Update existing alert
                existing_alert.updated_at = datetime.utcnow()
                existing_alert.annotations.update(alert.annotations)
                logger.debug(f"Updated existing alert: {alert.fingerprint}")
                return existing_alert.id
            
            # Add new alert
            self._active_alerts[alert.fingerprint] = alert
        
        logger.info(f"Fired alert: {alert.title} ({alert.severity.value})")
        
        # Route and send notifications
        await self._route_alert(alert)
        
        return alert.id
    
    async def resolve_alert(self, alert_id: str, resolution_message: str = "") -> bool:
        """Resolve an alert by ID."""
        with self._lock:
            alert = None
            for a in self._active_alerts.values():
                if a.id == alert_id:
                    alert = a
                    break
            
            if not alert:
                return False
            
            # Resolve the alert
            alert.resolve(resolution_message)
            
            # Move to history but keep in active alerts until filtered
            self._alert_history.append(alert)
        
        logger.info(f"Resolved alert: {alert.title}")
        
        # Send resolved notifications
        await self._send_resolved_notifications(alert)
        
        return True
    
    async def _route_alert(self, alert: Alert):
        """Route alert to appropriate channels."""
        matched_channels = set()
        
        with self._lock:
            routes = list(self._routes)
            available_channels = set(self._channels.keys())
        
        for route in routes:
            if not route.enabled:
                continue
            
            # Check severity filter
            if route.severity_filter and alert.severity not in route.severity_filter:
                continue
            
            # Check label matchers
            if self._matches_route(alert, route):
                matched_channels.update(route.channels)
        
        # If no routes match but we have channels, send to all channels (fallback behavior)
        if not matched_channels and available_channels:
            matched_channels = available_channels
            logger.debug(f"No routes configured, using fallback to send to all channels: {matched_channels}")
        
        # Send notifications
        if matched_channels:
            await self._send_notifications(alert, matched_channels)
        else:
            logger.warning(f"No matching routes for alert: {alert.title}")
    
    def _matches_route(self, alert: Alert, route: AlertRoute) -> bool:
        """Check if alert matches route conditions."""
        for label_key, expected_value in route.matchers.items():
            alert_value = alert.labels.get(label_key)
            
            if isinstance(expected_value, list):
                if alert_value not in expected_value:
                    return False
            else:
                if alert_value != expected_value:
                    return False
        
        return True
    
    async def _send_notifications(self, alert: Alert, channel_names: Set[str]):
        """Send notifications to specified channels."""
        # Check if alert should be aggregated
        if self.aggregator.should_aggregate(alert):
            self.aggregator.add_alert(alert)
            
            # Try to get aggregated alert
            aggregated = self.aggregator.get_aggregated_alert(alert.fingerprint)
            if aggregated:
                alert = aggregated
            else:
                return  # Wait for more alerts to aggregate
        
        with self._lock:
            channels = {name: self._channels[name] for name in channel_names if name in self._channels}
        
        for channel_name, channel in channels.items():
            if not channel.enabled:
                continue
            
            # Check rate limiting
            if not self.rate_limiter.can_send(channel_name):
                logger.warning(f"Rate limit exceeded for channel: {channel_name}")
                continue
            
            # Send notification
            try:
                success = await channel.send_notification(alert)
                if success:
                    self.rate_limiter.record_notification(channel_name)
            except Exception as e:
                logger.error(f"Failed to send notification via {channel_name}: {e}")
    
    async def _send_resolved_notifications(self, alert: Alert):
        """Send resolved notifications for an alert."""
        # Find channels that were notified about this alert
        matched_channels = set()
        
        with self._lock:
            routes = list(self._routes)
        
        for route in routes:
            if self._matches_route(alert, route):
                matched_channels.update(route.channels)
        
        with self._lock:
            channels = {name: self._channels[name] for name in matched_channels if name in self._channels}
        
        for channel_name, channel in channels.items():
            if not channel.enabled:
                continue
            
            notifier = self._notifiers.get(channel.type)
            if notifier:
                try:
                    await notifier.send_resolved(alert, channel)
                except Exception as e:
                    logger.error(f"Failed to send resolved notification via {channel_name}: {e}")
    
    async def evaluate_rules(self, metrics: Dict[str, Any]):
        """Evaluate alert rules against metrics."""
        with self._lock:
            rules = dict(self._rules)
        
        for rule_name, rule in rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Evaluate rule using the correct return type
                should_trigger, alert = rule.evaluate(metrics)
                
                if should_trigger and alert:
                    await self.fire_alert(alert)
                    
            except Exception as e:
                logger.error(f"Error evaluating rule {rule_name}: {e}")
                logger.error(f"Error evaluating rule {rule_name}: {e}")
    
    async def start_evaluation(self, evaluation_interval: float = 30.0):
        """Start periodic rule evaluation."""
        if self._evaluation_task:
            logger.warning("Rule evaluation already running")
            return
        
        self._evaluation_task = asyncio.create_task(
            self._evaluation_loop(evaluation_interval)
        )
        logger.info("Started alert rule evaluation")
    
    async def stop_evaluation(self):
        """Stop rule evaluation."""
        if self._evaluation_task:
            self._evaluation_task.cancel()
            try:
                await self._evaluation_task
            except asyncio.CancelledError:
                pass
            self._evaluation_task = None
            logger.info("Stopped alert rule evaluation")
    
    async def _evaluation_loop(self, interval: float):
        """Rule evaluation loop."""
        try:
            while True:
                await asyncio.sleep(interval)
                
                try:
                    # This would typically get metrics from your metrics system
                    # For now, we'll use a placeholder
                    metrics = await self._collect_metrics()
                    await self.evaluate_rules(metrics)
                except Exception as e:
                    logger.error(f"Error in evaluation loop: {e}")
        
        except asyncio.CancelledError:
            pass
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics for rule evaluation."""
        # This is a placeholder - integrate with your actual metrics system
        return {
            'timestamp': time.time(),
            'service_up': True,
            'error_rate': 0.1,
            'memory_usage_percent': 45.0,
            'cpu_usage_percent': 30.0,
            'gpu_usage_percent': 80.0
        }
    
    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        with self._lock:
            return [
                alert for alert in self._active_alerts.values()
                if alert.status != AlertStatus.RESOLVED
            ]
    
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """Get alert by ID."""
        with self._lock:
            for alert in self._active_alerts.values():
                if alert.id == alert_id:
                    return alert
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get alerting statistics."""
        with self._lock:
            all_alerts = list(self._active_alerts.values())
            active_alerts = [alert for alert in all_alerts if not alert.resolved]
            resolved_alerts = [alert for alert in all_alerts if alert.resolved]
            
            stats = {
                'total_alerts': len(all_alerts),
                'active_alerts': len(active_alerts),
                'resolved_alerts': len(resolved_alerts),
                'total_rules': len(self._rules),
                'total_channels': len(self._channels),
                'alerts_by_severity': {}
            }
            
            # Count by severity (for all alerts)
            for severity in AlertSeverity:
                count = len([alert for alert in all_alerts if alert.severity == severity])
                stats['alerts_by_severity'][severity.value] = count
            
            return stats
    
    def acknowledge_alert(self, fingerprint: str) -> bool:
        """Acknowledge an alert."""
        with self._lock:
            alert = self._active_alerts.get(fingerprint)
            if not alert:
                return False
            
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.updated_at = datetime.utcnow()
        
        logger.info(f"Acknowledged alert: {alert.title}")
        return True
    
    def suppress_alert(self, fingerprint: str) -> bool:
        """Suppress an alert."""
        with self._lock:
            alert = self._active_alerts.get(fingerprint)
            if not alert:
                return False
            
            alert.status = AlertStatus.SUPPRESSED
            alert.updated_at = datetime.utcnow()
        
        logger.info(f"Suppressed alert: {alert.title}")
        return True


def create_default_rules() -> List[AlertRule]:
    """Create default alert rules."""
    return [
        AlertRule(
            name="service_down",
            condition=lambda m: not m.get('service_up', True),
            severity=AlertSeverity.CRITICAL,
            title_template="Service Down",
            description_template="The inference service is not responding.",
            labels={'component': 'inference-service'},
            cooldown_seconds=300.0
        ),
        
        AlertRule(
            name="high_error_rate",
            condition=lambda m: m.get('error_rate', 0) > 0.05,  # 5%
            severity=AlertSeverity.WARNING,
            title_template="High Error Rate: {error_rate:.2%}",
            description_template="Error rate is {error_rate:.2%}, exceeding 5% threshold.",
            labels={'component': 'inference-service'},
            cooldown_seconds=600.0
        ),
        
        AlertRule(
            name="memory_leak",
            condition=lambda m: m.get('memory_usage_percent', 0) > 90,
            severity=AlertSeverity.CRITICAL,
            title_template="Memory Usage Critical: {memory_usage_percent:.1f}%",
            description_template="Memory usage is {memory_usage_percent:.1f}%, indicating potential memory leak.",
            labels={'component': 'system'},
            cooldown_seconds=300.0
        ),
        
        AlertRule(
            name="high_cpu_usage",
            condition=lambda m: m.get('cpu_usage_percent', 0) > 80,
            severity=AlertSeverity.WARNING,
            title_template="High CPU Usage: {cpu_usage_percent:.1f}%",
            description_template="CPU usage is {cpu_usage_percent:.1f}%, exceeding 80% threshold.",
            labels={'component': 'system'},
            cooldown_seconds=900.0
        ),
        
        AlertRule(
            name="gpu_overload",
            condition=lambda m: m.get('gpu_usage_percent', 0) > 95,
            severity=AlertSeverity.WARNING,
            title_template="GPU Overload: {gpu_usage_percent:.1f}%",
            description_template="GPU usage is {gpu_usage_percent:.1f}%, indicating overload conditions.",
            labels={'component': 'gpu'},
            cooldown_seconds=600.0
        )
    ]


# Global alert manager
_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get the global alert manager."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
