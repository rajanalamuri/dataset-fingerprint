"""mlfingerprint - lightweight, manifest-based dataset versioning for any modality.

import mlfingerprint as fp

v = fp.commit("./my_data", "s3://ml/ledgers/cars", message="added 3 batches")
ds = fp.load("s3://ml/ledgers/cars", "latest")
ds.files("train"); ds.verify(); ds.record_model("model-v1.0.0", {"acc": 0.94})
"""

__version__ = "0.1.0"

from .diff import Diff  # noqa: E402
from .ledger import (  # noqa: E402
    DatasetVersion,
    FingerprintError,
    Ledger,
    VerifyReport,
    commit,
    load,
    load_lock,
    record_training,
)
from .modality import register_extension, register_rule, register_sensor  # noqa: E402

__all__ = [
    "Ledger",
    "DatasetVersion",
    "VerifyReport",
    "Diff",
    "FingerprintError",
    "commit",
    "load",
    "load_lock",
    "record_training",
    "register_extension",
    "register_sensor",
    "register_rule",
    "__version__",
]
