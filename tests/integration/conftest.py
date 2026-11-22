"""Cloud emulators for integration tests.

S3     : moto's in-process server (pip install moto[server]) - always available when moto is installed.
Azure  : Azurite at 127.0.0.1:10000, or FP_AZURITE_CONN=<connection string>.
GCS    : fake-gcs-server at STORAGE_EMULATOR_HOST (default http://127.0.0.1:4443).

Anything unavailable is skipped, so `pytest -m integration` runs whatever is up.
CI starts all three as service containers (see .github/workflows/ci.yml).
"""

from __future__ import annotations

import os
import socket
import uuid
from urllib.parse import urlparse

import pytest

AZURITE_CONN = os.environ.get("FP_AZURITE_CONN") or (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;"
    "AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;"
    "BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)
GCS_HOST = os.environ.get("STORAGE_EMULATOR_HOST", "http://127.0.0.1:4443")
for h in ("127.0.0.1", "localhost"):
    for k in ("NO_PROXY", "no_proxy"):
        if h not in os.environ.get(k, ""):
            os.environ[k] = ",".join(x for x in (os.environ.get(k), h) if x)


def _port_open(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


class Cloud:
    """A bucket/container on one backend, plus raw helpers to set up fixtures."""

    def __init__(self, name, scheme, bucket, storage_options):
        self.name, self.scheme, self.bucket, self.so = name, scheme, bucket, storage_options
        import fsspec

        self.fs = fsspec.filesystem(scheme, **storage_options)

    def uri(self, *parts):
        return f"{self.scheme}://{self.bucket}/" + "/".join(parts)

    def put(self, key: str, data: bytes | str):
        self.fs.pipe(f"{self.bucket}/{key}", data.encode() if isinstance(data, str) else data)
        self.fs.invalidate_cache()

    def __repr__(self):
        return self.name


@pytest.fixture(scope="session")
def s3_server():
    moto_server = pytest.importorskip("moto.server")
    pytest.importorskip("s3fs")
    srv = moto_server.ThreadedMotoServer(ip_address="127.0.0.1", port=0)
    srv.start()
    host, port = srv.get_host_and_port()
    old = {k: os.environ.get(k) for k in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_DEFAULT_REGION", "AWS_ENDPOINT_URL")}
    os.environ.update(
        AWS_ACCESS_KEY_ID="testing", AWS_SECRET_ACCESS_KEY="testing", AWS_DEFAULT_REGION="us-east-1", AWS_ENDPOINT_URL=f"http://{host}:{port}"
    )
    yield f"http://{host}:{port}"
    srv.stop()
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _s3(s3_server):
    c = Cloud("s3", "s3", "it-" + uuid.uuid4().hex[:10], {"client_kwargs": {"endpoint_url": s3_server}})
    c.fs.mkdir(c.bucket)
    return c


def _azure():
    pytest.importorskip("adlfs")
    ep = urlparse([p.split("=", 1)[1] for p in AZURITE_CONN.split(";") if p.startswith("BlobEndpoint=")][0])
    if not _port_open(ep.hostname, ep.port or 443):
        pytest.skip("Azurite not running (npx azurite-blob --blobPort 10000)")
    c = Cloud("azure", "az", "it-" + uuid.uuid4().hex[:10], {"connection_string": AZURITE_CONN})
    c.fs.mkdir(c.bucket)
    return c


def _gcs():
    pytest.importorskip("gcsfs")
    u = urlparse(GCS_HOST)
    if not _port_open(u.hostname, u.port or 80):
        pytest.skip("fake-gcs-server not running (docker run -p 4443:4443 fsouza/fake-gcs-server -scheme http)")
    os.environ["STORAGE_EMULATOR_HOST"] = GCS_HOST
    c = Cloud("gcs", "gs", "it-" + uuid.uuid4().hex[:10], {"token": "anon", "endpoint_url": GCS_HOST, "project": "test"})
    c.fs.mkdir(c.bucket)
    return c


@pytest.fixture(params=["s3", "azure", "gcs"])
def cloud(request):
    if request.param == "s3":
        return _s3(request.getfixturevalue("s3_server"))
    return _azure() if request.param == "azure" else _gcs()


@pytest.fixture
def s3(s3_server):
    return _s3(s3_server)


@pytest.fixture
def azure():
    return _azure()
