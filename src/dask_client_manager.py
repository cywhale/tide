"""Compatibility shim for the removed Tide Dask client.

The Tide API uses direct Zarr reads and no longer connects to the shared
Dask scheduler. Keep these no-op functions for older internal imports while
the existing runtime manifest still carries this module during rollout.
"""


def get_dask_client(service_name: str | None = None) -> None:
    """Return no client; Tide no longer uses the shared Dask service."""
    return None


def close_dask_client(service_name: str | None = None) -> None:
    """Compatibility no-op; there is no Tide Dask client to close."""
    return None
