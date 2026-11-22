# Anonymous leakage audit

For datasets that cannot leave their environment. `anonymous_audit.py` runs dataset-fingerprint's
commit-time checks on each dataset in a throw-away ledger (deleted afterwards) and writes only
aggregate numbers:

| Written | Never written |
|---|---|
| a generic domain word you choose (`vision`, `radar`, …) | paths, bucket names, file or folder names |
| size bucket, modality counts, split type and sizes | class or label names, dataset names |
| leak rate, leaks by type, suspected (near-duplicate) rate | warning texts, examples, hashes, fingerprints |
| recording names shared by train and eval (a count) | anything from inside the files |
| record-level leak rate (text/tables), temporal overlap (yes/no) | |

Counts under 5 are shown as `<5` and the rest are rounded to two significant figures unless
`--exact` is passed. Failures are printed on your console and recorded only as an error type.

```bash
pip install "dataset-fingerprint[image]"     # add [s3] / [azure] / [gcs] for object stores
python anonymous_audit.py datasets.json --out audit
# review audit.json and audit.md, then share them
```

`datasets.json` lists `{"source": ..., "domain": ...}` objects plus any commit option
(`split`, `group_by`, `label_format`, `time_cutoffs`, `include`, `exclude`, …). See the script's docstring.
Only read access to the data is needed; nothing is written next to it.
