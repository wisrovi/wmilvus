"""Connection manager for Milvus connections."""

from typing import Any, Dict, Optional
from loguru import logger
from pymilvus import connections

from wmilvus.exceptions import ConnectionError


class ConnectionManager:
    """Manages connection lifecycles for Milvus server / cluster."""

    def __init__(
        self,
        alias: str = "default",
        host: str = "localhost",
        port: int = 19530,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        db_name: str = "default",
        **kwargs: Any,
    ) -> None:
        """Initialize connection configuration."""
        self.alias = alias
        self.host = host
        self.port = port
        self.uri = uri
        self.token = token
        self.user = user
        self.password = password
        self.db_name = db_name
        self.extra_kwargs = kwargs
        self._connected = False

    def connect(self) -> None:
        """Establish connection to Milvus."""
        try:
            conn_params: Dict[str, Any] = {"alias": self.alias, "db_name": self.db_name}
            if self.uri:
                conn_params["uri"] = self.uri
                if self.token:
                    conn_params["token"] = self.token
            else:
                conn_params["host"] = self.host
                conn_params["port"] = str(self.port)
                if self.user and self.password:
                    conn_params["user"] = self.user
                    conn_params["password"] = self.password
            
            conn_params.update(self.extra_kwargs)
            connections.connect(**conn_params)
            self._connected = True
            logger.info(f"Successfully connected to Milvus with alias '{self.alias}'")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {e}")
            raise ConnectionError(f"Failed to connect to Milvus: {e}") from e

    def disconnect(self) -> None:
        """Close connection to Milvus."""
        try:
            if self._connected:
                connections.disconnect(self.alias)
                self._connected = False
                logger.info(f"Disconnected from Milvus alias '{self.alias}'")
        except Exception as e:
            logger.warning(f"Error disconnecting alias '{self.alias}': {e}")

    @property
    def is_connected(self) -> bool:
        """Check if client is currently connected."""
        return self._connected
