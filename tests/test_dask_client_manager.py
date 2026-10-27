"""v0.3.2 shared-Dask client policy.

Missing, partial, or mismatched TLS settings, and any scheduler address that
is not a loopback IP, must never reach dask.distributed.Client (no plaintext
tcp:// fallback, no routable host); complete settings pass
Security(..., require_encryption=True). Network-free — Client and Security
are replaced with recorders.
"""
import pytest

import src.dask_client_manager as DCM

TLS_ENV = ("DASK_TLS_CA_FILE", "DASK_TLS_CLIENT_CERT", "DASK_TLS_CLIENT_KEY")
FULL = {"DASK_TLS_CA_FILE": "/x/ca.crt",
        "DASK_TLS_CLIENT_CERT": "/x/tide-client.crt",
        "DASK_TLS_CLIENT_KEY": "/x/tide-client.key"}
TLS_ADDR = "tls://127.0.0.1:8786"


class _Recorder:
    """Stands in for both the Client class and the Security class; each call
    is recorded and returns the recorder itself as the 'instance'."""
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self


@pytest.fixture
def fakes(monkeypatch):
    for k in TLS_ENV + ("DASK_SCHEDULER_ADDRESS", "TIDE_DASK_DISABLE"):
        monkeypatch.delenv(k, raising=False)
    client, security = _Recorder(), _Recorder()
    monkeypatch.setattr(DCM, "Client", client)
    monkeypatch.setattr(DCM, "Security", security)
    return client, security


def _manager(monkeypatch, env, address=None):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    return DCM.DaskClientManager(scheduler_address=address, service_name="tideapi-test")


@pytest.mark.parametrize("env,address", [
    ({}, None),                                   # nothing set: default tcp:// must not be used
    ({}, "tcp://127.0.0.1:8786"),
    ({}, TLS_ADDR),                               # tls:// without credentials
    ({"DASK_TLS_CA_FILE": "/x/ca.crt"}, TLS_ADDR),
    ({"DASK_TLS_CA_FILE": "/x/ca.crt", "DASK_TLS_CLIENT_CERT": "/x/c.crt"}, TLS_ADDR),
    ({"DASK_TLS_CLIENT_KEY": "/x/c.key"}, None),
    (FULL, "tcp://127.0.0.1:8786"),               # credentials with a plaintext address
    (FULL, None),                                 # credentials with the default tcp:// address
], ids=["all-unset-default", "all-unset-tcp", "tls-no-creds", "partial-ca",
        "partial-ca-cert", "partial-key-default", "creds-tcp", "creds-default"])
def test_incomplete_or_mismatched_tls_never_creates_client(fakes, monkeypatch, env, address):
    client, security = fakes
    m = _manager(monkeypatch, env, address)
    assert m.get_client() is None
    assert m.client is None
    assert client.calls == []
    assert security.calls == []


@pytest.mark.parametrize("address", [
    "tls://192.168.2.24:8786",                   # VM24's routable address
    "tls://10.0.0.5:8786",
    "tls://0.0.0.0:8786",                        # unspecified, not loopback
    "tls://[2001:db8::1]:8786",
    "tls://localhost:8786",                      # hostnames rejected, even localhost
    "tls://scheduler.example:8786",
    "tls://evil.example@127.0.0.1:8786",         # userinfo trick: host is not an IP
])
def test_non_loopback_scheduler_never_creates_client(fakes, monkeypatch, address):
    client, security = fakes
    m = _manager(monkeypatch, FULL, address)
    assert m.get_client() is None
    assert client.calls == []
    assert security.calls == []


@pytest.mark.parametrize("address", ["tls://127.0.0.1:8786", "tls://[::1]:8786"])
def test_loopback_ip_scheduler_is_accepted(fakes, monkeypatch, address):
    client, _ = fakes
    m = _manager(monkeypatch, FULL, address)
    assert m.get_client() is client
    assert client.calls[0][0] == (address,)


def test_complete_tls_passes_security_with_require_encryption(fakes, monkeypatch):
    client, security = fakes
    m = _manager(monkeypatch, FULL, TLS_ADDR)
    assert m.get_client() is client
    (_, skw), = security.calls
    assert skw == {"tls_ca_file": "/x/ca.crt",
                   "tls_client_cert": "/x/tide-client.crt",
                   "tls_client_key": "/x/tide-client.key",
                   "require_encryption": True}
    (cargs, ckw), = client.calls
    assert cargs == (TLS_ADDR,)
    assert ckw["security"] is security


def test_scheduler_address_from_env(fakes, monkeypatch):
    client, _ = fakes
    monkeypatch.setenv("DASK_SCHEDULER_ADDRESS", TLS_ADDR)
    m = _manager(monkeypatch, FULL)
    assert m.get_client() is client
    assert client.calls[0][0] == (TLS_ADDR,)


def test_dask_disable_skips_connect_even_with_tls(fakes, monkeypatch):
    client, _ = fakes
    monkeypatch.setenv("TIDE_DASK_DISABLE", "1")
    m = _manager(monkeypatch, FULL, TLS_ADDR)
    assert m.get_client() is None
    assert client.calls == []
