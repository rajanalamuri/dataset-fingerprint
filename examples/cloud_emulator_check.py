"""Smoke test against local cloud emulators (no real cloud account needed).

pip install "dataset-fingerprint[s3,azure]" "moto[server]" boto3 azure-storage-blob
moto_server -p 5555 &                      # S3
npx azurite-blob --blobPort 10000 &        # Azure Blob
NO_PROXY=127.0.0.1 python examples/cloud_emulator_check.py
"""

import os

os.environ.setdefault("AWS_ACCESS_KEY_ID", "x")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "x")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")
os.environ.setdefault("AWS_ENDPOINT_URL", "http://127.0.0.1:5555")

import boto3  # noqa: E402
from azure.storage.blob import BlobServiceClient  # noqa: E402

import dataset_fingerprint as fp  # noqa: E402

AZURITE = (
    "DefaultEndpointsProtocol=http;AccountName=devstoreaccount1;AccountKey=Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6"
    "IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw==;BlobEndpoint=http://127.0.0.1:10000/devstoreaccount1;"
)

s3 = boto3.client("s3", endpoint_url=os.environ["AWS_ENDPOINT_URL"])
for b in ("fp-raw", "fp-ledger"):
    try:
        s3.create_bucket(Bucket=b)
    except Exception:
        pass
for run in range(24):
    for f in range(3):
        base = f"drives/2026-04-{run % 28 + 1:02d}_run{run}"
        s3.put_object(Bucket="fp-raw", Key=f"{base}/radar/{f:06d}.bin", Body=os.urandom(32))
        s3.put_object(Bucket="fp-raw", Key=f"{base}/camera/{f:06d}.jpg", Body=os.urandom(32))

L = "s3://fp-ledger/drives"
v = fp.commit("s3://fp-raw/drives", L, name="drives", split="temporal", mode="cas", quiet=True)
print("S3 temporal+cas :", v.version, v["split_strategy"]["time_ranges"]["test"], "objects:", v["source"]["objects_written"])
lg = fp.Ledger(L)
lg.tag("latest", "prod")
print("S3 verify       :", lg.get("prod").verify().render(), "| quick:", lg.get("prod").verify(quick=True).ok)
print("S3 checkout     :", v.checkout("s3://fp-ledger/checkout-test", split="test"))

ref = fp.commit("s3://fp-raw/drives", "s3://fp-ledger/ref", quiet=True)  # reference mode, zero copy
s3.put_object(Bucket="fp-raw", Key="drives/2026-04-01_run0/radar/000000.bin", Body=b"tampered")
rep = fp.load("s3://fp-ledger/ref").verify()
print("S3 tamper (ref) : caught =", not rep.ok, rep.modified)
print("S3 cas snapshot : still intact =", v.verify().ok)

svc = BlobServiceClient.from_connection_string(AZURITE)
try:
    svc.create_container("ledgers")
except Exception:
    pass
so = {"connection_string": AZURITE}
d = fp.commit(
    "s3://fp-raw/drives",
    "az://ledgers/drives-aug",
    name="drives-aug",
    source_storage_options={},
    storage_options=so,
    parents=[v],
    mode="copy",
    quiet=True,
)
print(
    "S3 -> Azure     :",
    d.version,
    "parent:",
    d.trace()["parents"][0]["dataset_version"],
    "verify:",
    fp.load("az://ledgers/drives-aug", storage_options=so).verify().ok,
)
print(
    "abfs://         :",
    fp.commit("s3://fp-raw/drives", "abfs://ledgers/abfs-test", source_storage_options={}, storage_options=so, quiet=True).version,
)
