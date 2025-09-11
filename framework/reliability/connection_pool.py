"""
Connection Pooling System

Provides connection pooling for database/cache connections with proper lifecycle management.
Supports multiple connection types and automatic cleanup.
"""

import asyncio
import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Union, Generic, TypeVar
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
import weakref

logger = logging.getLogger(__name__)

T = TypeVar('T')


class ConnectionState(Enum):
    """Connection states."""
    IDLE = "idle"
    ACTIVE = "active"
    BROKEN = "broken"
    CLOSED = "closed"


@dataclass
class PoolConfig:
    """Connection pool configuration."""
    min_size: int = 1                    # Minimum pool size
    max_size: int = 10                   # Maximum pool size
    max_idle_time: float = 300.0         # Max idle time before closing (seconds)
    max_lifetime: float = 3600.0         # Max connection lifetime (seconds)
    connection_timeout: float = 30.0     # Connection creation timeout
    health_check_interval: float = 60.0  # Health check interval
    retry_attempts: int = 3              # Connection retry attempts
    retry_delay: float = 1.0             # Delay between retry attempts
    
    # Pool behavior
    validate_on_borrow: bool = True      # Validate connection before use
    validate_on_return: bool = False     # Validate connection after use
    test_on_idle: bool = True           # Test connections during idle
    preemptive_validation: bool = True   # Validate before max_idle_time
    
    # Cleanup settings
    cleanup_interval: float = 30.0      # Pool cleanup interval
    force_close_timeout: float = 5.0    # Force close timeout
    enable_metrics: bool = True         # Enable pool metrics


@dataclass
class ConnectionInfo:
    """Information about a pooled connection."""
    connection: Any
    created_at: float
    last_used_at: float
    use_count: int = 0
    state: ConnectionState = ConnectionState.IDLE
    
    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get connection idle time in seconds."""
        return time.time() - self.last_used_at
    
    def mark_used(self):
        """Mark connection as used."""
        self.last_used_at = time.time()
        self.use_count += 1
        self.state = ConnectionState.ACTIVE
    
    def mark_idle(self):
        """Mark connection as idle."""
        self.state = ConnectionState.IDLE


@dataclass
class PoolMetrics:
    """Connection pool metrics."""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    broken_connections: int = 0
    
    total_borrowed: int = 0
    total_returned: int = 0
    total_created: int = 0
    total_destroyed: int = 0
    
    borrow_wait_time_total: float = 0.0
    borrow_wait_count: int = 0


# Alias for compatibility
PoolStats = PoolMetrics


class PoolConnection:
    """Wrapper for pooled connections with metadata."""
    
    def __init__(self, connection: Any):
        self.connection = connection
        self.created_at = time.time()
        self.last_used = self.created_at
        self.use_count = 0
        self.in_use = False
    
    def checkout(self):
        """Checkout the connection for use."""
        self.in_use = True
        self.last_used = time.time()
        self.use_count += 1
    
    def checkin(self):
        """Check in the connection after use."""
        self.in_use = False
    
    @property
    def age(self) -> float:
        """Get connection age in seconds."""
        return time.time() - self.created_at
    
    @property
    def idle_time(self) -> float:
        """Get idle time since last use."""
        return time.time() - self.last_used


class ConnectionFactory(ABC):
    """Abstract base class for connection factories."""
    
    @abstractmethod
    async def create_connection(self) -> Any:
        """Create a new connection."""
        pass
    
    @abstractmethod
    async def validate_connection(self, connection: Any) -> bool:
        """Validate that a connection is still healthy."""
        pass
    
    @abstractmethod
    async def close_connection(self, connection: Any):
        """Close a connection."""
        pass


class RedisConnectionFactory(ConnectionFactory):
    """Factory for Redis connections."""
    
    def __init__(self, host: str = "localhost", port: int = 6379, db: int = 0, **kwargs):
        self.host = host
        self.port = port
        self.db = db
        self.connection_kwargs = kwargs
    
    async def create_connection(self) -> Any:
        """Create Redis connection."""
        try:
            import aioredis
            return await aioredis.from_url(
                f"redis://{self.host}:{self.port}/{self.db}",
                **self.connection_kwargs
            )
        except ImportError:
            raise ImportError("aioredis required for Redis connections")
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate Redis connection."""
        try:
            await connection.ping()
            return True
        except Exception:
            return False
    
    async def close_connection(self, connection: Any):
        """Close Redis connection."""
        try:
            await connection.close()
        except Exception:
            pass


class DatabaseConnectionFactory(ConnectionFactory):
    """Factory for database connections."""
    
    def __init__(self, database_url: str, **kwargs):
        self.database_url = database_url
        self.connection_kwargs = kwargs
    
    async def create_connection(self) -> Any:
        """Create database connection."""
        try:
            import asyncpg
            return await asyncpg.connect(self.database_url, **self.connection_kwargs)
        except ImportError:
            raise ImportError("asyncpg required for PostgreSQL connections")
    
    async def validate_connection(self, connection: Any) -> bool:
        """Validate database connection."""
        try:
            await connection.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    async def close_connection(self, connection: Any):
        """Close database connection."""
        try:
            await connection.close()
        except Exception:
            pass
    
    connection_errors: int = 0
    validation_failures: int = 0
    
    @property
    def average_borrow_wait_time(self) -> float:
        """Calculate average borrow wait time."""
        if self.borrow_wait_count == 0:
            return 0.0
        return self.borrow_wait_time_total / self.borrow_wait_count


class ConnectionFactory(ABC, Generic[T]):
    """Abstract factory for creating connections."""
    
    @abstractmethod
    async def create_connection(self) -> T:
        """Create a new connection."""
        pass
    
    @abstractmethod
    async def validate_connection(self, connection: T) -> bool:
        """Validate if a connection is still usable."""
        pass
    
    @abstractmethod
    async def close_connection(self, connection: T):
        """Close a connection."""
        pass
    
    @abstractmethod
    def is_connection_error(self, error: Exception) -> bool:
        """Check if an error indicates a connection problem."""
        pass


class AsyncConnectionPool(Generic[T]):
    """
    Async connection pool with lifecycle management.
    
    Provides:
    - Automatic connection creation and cleanup
    - Connection validation and health checks
    - Metrics and monitoring
    - Graceful shutdown support
    """
    
    def __init__(self, 
                 factory: ConnectionFactory[T], 
                 config: Optional[PoolConfig] = None):
        self.factory = factory
        self.config = config or PoolConfig()
        
        # Connection management
        self._connections: Dict[int, ConnectionInfo] = {}
        self._available: asyncio.Queue = asyncio.Queue(maxsize=self.config.max_size)
        self._waiters: List[asyncio.Future] = []
        
        # State management
        self._closed = False
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self.metrics = PoolMetrics()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Thread safety
        self._lock = asyncio.Lock()
        
        logger.info(f"Connection pool created with config: min={self.config.min_size}, "
                   f"max={self.config.max_size}")
    
    async def start(self):
        """Start the connection pool."""
        async with self._lock:
            if self._closed:
                raise RuntimeError("Cannot start a closed pool")
            
            # Create minimum connections
            for _ in range(self.config.min_size):
                try:
                    await self._create_connection()
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")
            
            # Start background tasks
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._health_check_task = asyncio.create_task(self._health_check_loop())
            
            logger.info(f"Connection pool started with {len(self._connections)} connections")
    
    async def close(self):
        """Close the connection pool."""
        async with self._lock:
            if self._closed:
                return
            
            self._closed = True
            self._shutdown_event.set()
            
            # Cancel background tasks
            if self._cleanup_task:
                self._cleanup_task.cancel()
            if self._health_check_task:
                self._health_check_task.cancel()
            
            # Wait for tasks to finish
            tasks = [t for t in [self._cleanup_task, self._health_check_task] if t]
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Cancel waiting requests
            for waiter in self._waiters:
                if not waiter.done():
                    waiter.cancel()
            self._waiters.clear()
            
            # Close all connections
            for conn_info in list(self._connections.values()):
                await self._destroy_connection(conn_info)
            
            self._connections.clear()
            
            # Clear available queue
            while not self._available.empty():
                try:
                    self._available.get_nowait()
                except asyncio.QueueEmpty:
                    break
            
            logger.info("Connection pool closed")
    
    @asynccontextmanager
    async def get_connection(self):
        """Get a connection from the pool (context manager)."""
        connection = await self.borrow_connection()
        try:
            yield connection
        finally:
            await self.return_connection(connection)
    
    async def borrow_connection(self) -> T:
        """Borrow a connection from the pool."""
        if self._closed:
            raise RuntimeError("Pool is closed")
        
        borrow_start = time.time()
        
        try:
            # Try to get an available connection
            conn_info = await asyncio.wait_for(
                self._get_available_connection(),
                timeout=self.config.connection_timeout
            )
            
            # Update metrics
            wait_time = time.time() - borrow_start
            self.metrics.borrow_wait_time_total += wait_time
            self.metrics.borrow_wait_count += 1
            self.metrics.total_borrowed += 1
            
            # Mark as used
            conn_info.mark_used()
            
            logger.debug(f"Borrowed connection (id={id(conn_info.connection)}, "
                        f"wait_time={wait_time*1000:.1f}ms)")
            
            return conn_info.connection
            
        except asyncio.TimeoutError:
            logger.error(f"Connection borrow timeout after {self.config.connection_timeout}s")
            raise
        except Exception as e:
            logger.error(f"Failed to borrow connection: {e}")
            raise
    
    async def return_connection(self, connection: T):
        """Return a connection to the pool."""
        if self._closed:
            return
        
        conn_id = id(connection)
        
        async with self._lock:
            conn_info = self._connections.get(conn_id)
            if not conn_info:
                logger.warning(f"Attempted to return unknown connection (id={conn_id})")
                return
            
            # Validate connection if configured
            is_valid = True
            if self.config.validate_on_return:
                try:
                    is_valid = await self.factory.validate_connection(connection)
                except Exception as e:
                    logger.warning(f"Connection validation failed on return: {e}")
                    is_valid = False
                    self.metrics.validation_failures += 1
            
            if not is_valid:
                conn_info.state = ConnectionState.BROKEN
                await self._destroy_connection(conn_info)
                logger.debug(f"Destroyed invalid connection on return (id={conn_id})")
            else:
                conn_info.mark_idle()
                await self._available.put(conn_info)
                logger.debug(f"Returned connection to pool (id={conn_id})")
            
            self.metrics.total_returned += 1
    
    async def _get_available_connection(self) -> ConnectionInfo:
        """Get an available connection, creating one if needed."""
        while True:
            try:
                # Try to get from available queue
                conn_info = self._available.get_nowait()
                
                # Validate if configured
                if self.config.validate_on_borrow:
                    try:
                        is_valid = await self.factory.validate_connection(conn_info.connection)
                    except Exception as e:
                        logger.warning(f"Connection validation failed on borrow: {e}")
                        is_valid = False
                        self.metrics.validation_failures += 1
                    
                    if not is_valid:
                        conn_info.state = ConnectionState.BROKEN
                        await self._destroy_connection(conn_info)
                        continue  # Try next connection
                
                return conn_info
                
            except asyncio.QueueEmpty:
                # No available connections, try to create one
                if len(self._connections) < self.config.max_size:
                    try:
                        conn_info = await self._create_connection()
                        return conn_info
                    except Exception as e:
                        logger.error(f"Failed to create new connection: {e}")
                        self.metrics.connection_errors += 1
                
                # Wait for a connection to become available
                waiter = asyncio.Future()
                self._waiters.append(waiter)
                
                try:
                    await waiter
                    # Loop again to try getting a connection
                except asyncio.CancelledError:
                    raise
                finally:
                    if waiter in self._waiters:
                        self._waiters.remove(waiter)
    
    async def _create_connection(self) -> ConnectionInfo:
        """Create a new connection."""
        try:
            connection = await self.factory.create_connection()
            
            conn_info = ConnectionInfo(
                connection=connection,
                created_at=time.time(),
                last_used_at=time.time(),
                state=ConnectionState.IDLE
            )
            
            self._connections[id(connection)] = conn_info
            self.metrics.total_created += 1
            self.metrics.total_connections = len(self._connections)
            
            logger.debug(f"Created new connection (id={id(connection)})")
            
            return conn_info
            
        except Exception as e:
            logger.error(f"Failed to create connection: {e}")
            self.metrics.connection_errors += 1
            raise
    
    async def _destroy_connection(self, conn_info: ConnectionInfo):
        """Destroy a connection."""
        conn_id = id(conn_info.connection)
        
        try:
            await self.factory.close_connection(conn_info.connection)
            logger.debug(f"Closed connection (id={conn_id})")
            
        except Exception as e:
            logger.warning(f"Error closing connection (id={conn_id}): {e}")
        
        finally:
            # Remove from tracking
            if conn_id in self._connections:
                del self._connections[conn_id]
            
            conn_info.state = ConnectionState.CLOSED
            
            self.metrics.total_destroyed += 1
            self.metrics.total_connections = len(self._connections)
            
            # Notify waiters
            if self._waiters and not self._closed:
                waiter = self._waiters.pop(0)
                if not waiter.done():
                    waiter.set_result(None)
    
    async def _cleanup_loop(self):
        """Background task for connection cleanup."""
        try:
            while not self._closed:
                await asyncio.sleep(self.config.cleanup_interval)
                
                if self._closed:
                    break
                
                await self._cleanup_connections()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_connections(self):
        """Clean up old and unused connections."""
        async with self._lock:
            current_time = time.time()
            connections_to_destroy = []
            
            for conn_info in self._connections.values():
                should_destroy = False
                
                # Check max lifetime
                if conn_info.age > self.config.max_lifetime:
                    should_destroy = True
                    logger.debug(f"Connection exceeded max lifetime: {conn_info.age:.1f}s")
                
                # Check max idle time (only for idle connections)
                elif (conn_info.state == ConnectionState.IDLE and 
                      conn_info.idle_time > self.config.max_idle_time):
                    # Don't destroy if it would go below minimum
                    active_count = len([c for c in self._connections.values() 
                                      if c.state != ConnectionState.BROKEN])
                    if active_count > self.config.min_size:
                        should_destroy = True
                        logger.debug(f"Connection exceeded max idle time: {conn_info.idle_time:.1f}s")
                
                # Check if broken
                elif conn_info.state == ConnectionState.BROKEN:
                    should_destroy = True
                    logger.debug("Cleaning up broken connection")
                
                if should_destroy:
                    connections_to_destroy.append(conn_info)
            
            # Destroy connections that need cleanup
            for conn_info in connections_to_destroy:
                await self._destroy_connection(conn_info)
    
    async def _health_check_loop(self):
        """Background task for connection health checks."""
        try:
            while not self._closed:
                await asyncio.sleep(self.config.health_check_interval)
                
                if self._closed:
                    break
                
                await self._health_check_connections()
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Error in health check loop: {e}")
    
    async def _health_check_connections(self):
        """Perform health checks on idle connections."""
        if not self.config.test_on_idle:
            return
        
        async with self._lock:
            idle_connections = [
                conn_info for conn_info in self._connections.values()
                if conn_info.state == ConnectionState.IDLE
            ]
            
            for conn_info in idle_connections:
                try:
                    is_valid = await self.factory.validate_connection(conn_info.connection)
                    if not is_valid:
                        conn_info.state = ConnectionState.BROKEN
                        logger.debug(f"Health check failed for connection {id(conn_info.connection)}")
                        self.metrics.validation_failures += 1
                
                except Exception as e:
                    logger.warning(f"Health check error for connection {id(conn_info.connection)}: {e}")
                    conn_info.state = ConnectionState.BROKEN
                    self.metrics.validation_failures += 1
    
    def get_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics."""
        # Update current metrics
        self.metrics.active_connections = sum(
            1 for c in self._connections.values() if c.state == ConnectionState.ACTIVE
        )
        self.metrics.idle_connections = sum(
            1 for c in self._connections.values() if c.state == ConnectionState.IDLE
        )
        self.metrics.broken_connections = sum(
            1 for c in self._connections.values() if c.state == ConnectionState.BROKEN
        )
        
        return {
            "pool_size": len(self._connections),
            "available_connections": self._available.qsize(),
            "active_connections": self.metrics.active_connections,
            "idle_connections": self.metrics.idle_connections,
            "broken_connections": self.metrics.broken_connections,
            "waiting_requests": len(self._waiters),
            "metrics": {
                "total_borrowed": self.metrics.total_borrowed,
                "total_returned": self.metrics.total_returned,
                "total_created": self.metrics.total_created,
                "total_destroyed": self.metrics.total_destroyed,
                "connection_errors": self.metrics.connection_errors,
                "validation_failures": self.metrics.validation_failures,
                "average_borrow_wait_time_ms": self.metrics.average_borrow_wait_time * 1000
            },
            "config": {
                "min_size": self.config.min_size,
                "max_size": self.config.max_size,
                "max_idle_time": self.config.max_idle_time,
                "max_lifetime": self.config.max_lifetime
            },
            "closed": self._closed
        }


class ConnectionPoolManager:
    """
    Manager for multiple connection pools.
    
    Provides centralized management of different connection pools
    (database, cache, etc.) with unified monitoring and control.
    """
    
    def __init__(self):
        self._pools: Dict[str, AsyncConnectionPool] = {}
        self._lock = asyncio.Lock()
        logger.info("Connection pool manager initialized")
    
    async def create_pool(self, 
                         name: str, 
                         factory: ConnectionFactory, 
                         config: Optional[PoolConfig] = None) -> AsyncConnectionPool:
        """Create and register a new connection pool."""
        async with self._lock:
            if name in self._pools:
                raise ValueError(f"Pool '{name}' already exists")
            
            pool = AsyncConnectionPool(factory, config)
            await pool.start()
            
            self._pools[name] = pool
            logger.info(f"Created connection pool: {name}")
            
            return pool
    
    async def get_pool(self, name: str) -> Optional[AsyncConnectionPool]:
        """Get an existing connection pool."""
        async with self._lock:
            return self._pools.get(name)
    
    async def remove_pool(self, name: str) -> bool:
        """Remove and close a connection pool."""
        async with self._lock:
            if name not in self._pools:
                return False
            
            pool = self._pools[name]
            await pool.close()
            del self._pools[name]
            
            logger.info(f"Removed connection pool: {name}")
            return True
    
    async def close_all(self):
        """Close all connection pools."""
        async with self._lock:
            for name, pool in self._pools.items():
                try:
                    await pool.close()
                    logger.info(f"Closed connection pool: {name}")
                except Exception as e:
                    logger.error(f"Error closing pool {name}: {e}")
            
            self._pools.clear()
            logger.info("All connection pools closed")
    
    def get_all_stats(self) -> Dict[str, Any]:
        """Get statistics for all connection pools."""
        return {
            "pools": {
                name: pool.get_stats() 
                for name, pool in self._pools.items()
            },
            "total_pools": len(self._pools)
        }


# Global connection pool manager
_connection_pool_manager = None


def get_connection_pool_manager() -> ConnectionPoolManager:
    """Get the global connection pool manager."""
    global _connection_pool_manager
    if _connection_pool_manager is None:
        _connection_pool_manager = ConnectionPoolManager()
    return _connection_pool_manager


# Example connection factory implementations
class RedisConnectionFactory(ConnectionFactory):
    """Connection factory for Redis connections."""
    
    def __init__(self, url: str, **kwargs):
        self.url = url
        self.kwargs = kwargs
    
    async def create_connection(self):
        """Create a Redis connection."""
        try:
            import aioredis
            return await aioredis.from_url(self.url, **self.kwargs)
        except ImportError:
            raise ImportError("aioredis is required for Redis connection pooling")
    
    async def validate_connection(self, connection) -> bool:
        """Validate Redis connection."""
        try:
            await connection.ping()
            return True
        except Exception:
            return False
    
    async def close_connection(self, connection):
        """Close Redis connection."""
        try:
            await connection.close()
        except Exception:
            pass
    
    def is_connection_error(self, error: Exception) -> bool:
        """Check if error indicates connection problem."""
        # This would need to be customized based on Redis error types
        return True


class DatabaseConnectionFactory(ConnectionFactory):
    """Connection factory for database connections."""
    
    def __init__(self, connection_string: str, **kwargs):
        self.connection_string = connection_string
        self.kwargs = kwargs
    
    async def create_connection(self):
        """Create a database connection."""
        # This is a placeholder - actual implementation would depend on database type
        # For example, with asyncpg for PostgreSQL:
        # import asyncpg
        # return await asyncpg.connect(self.connection_string, **self.kwargs)
        raise NotImplementedError("Database connection factory needs specific implementation")
    
    async def validate_connection(self, connection) -> bool:
        """Validate database connection."""
        try:
            # Example validation query
            # await connection.fetchval("SELECT 1")
            return True
        except Exception:
            return False
    
    async def close_connection(self, connection):
        """Close database connection."""
        try:
            await connection.close()
        except Exception:
            pass
    
    def is_connection_error(self, error: Exception) -> bool:
        """Check if error indicates connection problem."""
        # This would need to be customized based on database error types
        return True
