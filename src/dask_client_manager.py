from dask.distributed import Client
from functools import partial
import uuid

class DaskClientManager:
    """
    Manages Dask client connections with unique naming for different FastAPI services
    """
    def __init__(self, scheduler_address: str = 'tcp://localhost:8786', service_name: str = None):
        self.scheduler_address = scheduler_address
        # Generate a unique service identifier if none provided
        self.service_name = service_name or f"service-{uuid.uuid4().hex[:8]}"
        self.client = None

    def get_client(self):
        """
        Returns a configured Dask client with unique naming scheme
        """
        if self.client is None:
            self.client = Client(
                self.scheduler_address,
                name=self.service_name,
                set_as_default=True
            )
            
            # Configure unique key prefix for this service
            def key_prefix(key):
                return f"{self.service_name}-{key}"
            
            # Patch the client's key generation
            self.client._graph_key_prefix = key_prefix
            
        return self.client

    def close(self):
        """
        Properly close the client connection
        """
        if self.client is not None:
            self.client.close()
            self.client = None

# Usage in each FastAPI app's main.py
def get_dask_client(service_name: str):
    manager = DaskClientManager(service_name=service_name)
    return manager.get_client()
