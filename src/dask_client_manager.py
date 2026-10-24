import logging
import os
import threading
import uuid

from dask.distributed import Client
from distributed.security import Security

class DaskClientManager:
    """
    Manages Dask client connections with unique naming for different FastAPI services
    """
    def __init__(self, scheduler_address: str = None, service_name: str = None):
        env_address = os.getenv("DASK_SCHEDULER_ADDRESS")
        self.scheduler_address = scheduler_address or env_address or "tcp://localhost:8786"
        # Generate a unique service identifier if none provided
        self.service_name = service_name or f"service-{uuid.uuid4().hex[:8]}"
        self.client = None
        self._connect_lock = threading.Lock()
        self._log = logging.getLogger(__name__)

    def _connect(self):
        # Fast path (round 23 F3): skip the connect entirely when dask is
        # disabled (tests / single-process runs), avoiding the multi-second
        # connect timeout when no scheduler is reachable. The data paths do
        # not require a dask client.
        if os.getenv("TIDE_DASK_DISABLE", "").strip().lower() in ("1", "true", "yes"):
            self._log.info("Dask disabled via TIDE_DASK_DISABLE; no client.")
            self.client = None
            return None
        try:
            tls_settings = (
                os.getenv("DASK_TLS_CA_FILE"),
                os.getenv("DASK_TLS_CLIENT_CERT"),
                os.getenv("DASK_TLS_CLIENT_KEY"),
            )
            if any(tls_settings) and not all(tls_settings):
                raise ValueError("DASK TLS requires CA, client certificate, and client key")
            if self.scheduler_address.startswith("tls://") and not all(tls_settings):
                raise ValueError("TLS scheduler configured without complete DASK TLS credentials")
            if all(tls_settings) and not self.scheduler_address.startswith("tls://"):
                raise ValueError("DASK TLS credentials configured for a non-TLS scheduler address")
            security = (
                Security(
                    tls_ca_file=tls_settings[0],
                    tls_client_cert=tls_settings[1],
                    tls_client_key=tls_settings[2],
                    require_encryption=True,
                )
                if all(tls_settings)
                else None
            )
            self.client = Client(
                self.scheduler_address,
                name=self.service_name,
                set_as_default=True,
                security=security,
                timeout="5s",   # fail fast if no scheduler (was ~30s default)
            )

            # Configure unique key prefix for this service
            def key_prefix(key):
                return f"{self.service_name}-{key}"

            # Patch the client's key generation
            self.client._graph_key_prefix = key_prefix
            return self.client
        except Exception as exc:
            self._log.warning(
                "Dask client connection failed for %s (%s): %s",
                self.service_name,
                self.scheduler_address,
                exc,
            )
            self.client = None
            return None

    def _is_healthy(self) -> bool:
        if self.client is None:
            return False
        if self.client.status in ("closed", "closing"):
            return False
        try:
            self.client.scheduler_info()
        except Exception:
            return False
        return True

    def get_client(self):
        """
        Returns a configured Dask client with unique naming scheme
        """
        with self._connect_lock:
            if self._is_healthy():
                return self.client
            if self.client is not None:
                self.close()
            return self._connect()

    def close(self):
        """
        Properly close the client connection
        """
        if self.client is not None:
            self.client.close()
            self.client = None

_MANAGERS = {}

def _get_manager(service_name: str):
    manager = _MANAGERS.get(service_name)
    if manager is None:
        manager = DaskClientManager(service_name=service_name)
        _MANAGERS[service_name] = manager
    return manager

# Usage in each FastAPI app's main.py
def get_dask_client(service_name: str):
    manager = _get_manager(service_name)
    return manager.get_client()

def close_dask_client(service_name: str):
    manager = _MANAGERS.get(service_name)
    if manager is not None:
        manager.close()
