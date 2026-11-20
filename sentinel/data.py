"""Versioned NASA FD001 acquisition, parsing and integrity checks."""
from __future__ import annotations
import hashlib
from pathlib import Path
from urllib.request import urlopen, Request
from zipfile import ZipFile
import numpy as np
import pandas as pd

SOURCE_URL = "https://zenodo.org/records/15346912/files/CMAPSSData.zip?download=1"
SOURCE_MD5 = "79a22f36e80606c69d0e9e4da5bb2b7a"
COLS = ["unit", "cycle"] + [f"setting_{i}" for i in range(1, 4)] + [f"sensor_{i}" for i in range(1, 22)]

def download_dataset(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "CMAPSSData.zip"
    if not archive.exists() or hashlib.md5(archive.read_bytes()).hexdigest() != SOURCE_MD5:
        req = Request(SOURCE_URL, headers={"User-Agent": "MUSITU-AXIOM-Sentinel/0.1"})
        chunks, size = [], 0
        with urlopen(req, timeout=90) as stream:
            while part := stream.read(1024 * 1024):
                size += len(part)
                if size > 100 * 1024 * 1024:
                    raise ValueError("Downloaded file unexpectedly exceeds 100MB")
                chunks.append(part)
        payload = b"".join(chunks)
        if hashlib.md5(payload).hexdigest() != SOURCE_MD5:
            raise ValueError("Source MD5 mismatch; do not use unverified data")
        archive.write_bytes(payload)
    with ZipFile(archive) as z:
        members = {Path(name).name: name for name in z.namelist()}
        required = ("train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt")
        if not all(name in members for name in required):
            raise ValueError("Source archive lacks NASA FD001 benchmark files")
        for name in required:
            (root / name).write_bytes(z.read(members[name]))
    return root

def read_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="python").dropna(axis=1, how="all")
    if df.shape[1] != 26:
        raise ValueError(f"{path}: expected 26 columns, found {df.shape[1]}")
    df.columns = COLS
    if df.isna().any().any() or not np.isfinite(df.to_numpy(dtype=float)).all():
        raise ValueError(f"{path}: missing or non-finite values")
    df[["unit", "cycle"]] = df[["unit", "cycle"]].astype(int)
    if df.duplicated(["unit", "cycle"]).any():
        raise ValueError("Duplicate engine/cycle rows")
    for _, group in df.groupby("unit"):
        if not np.array_equal(group["cycle"].to_numpy(), np.arange(1, len(group) + 1)):
            raise ValueError("Cycles are non-contiguous or not sorted")
    return df

def load_fd001(root: Path):
    train = read_table(root / "train_FD001.txt")
    test = read_table(root / "test_FD001.txt")
    labels = pd.read_csv(root / "RUL_FD001.txt", sep=r"\s+", header=None).dropna(axis=1, how="all")
    test_ids = sorted(test.unit.unique())
    if labels.shape[1] != 1 or len(labels) != len(test_ids):
        raise ValueError("Official RUL counts do not match test engine count")
    return train, test, dict(zip(test_ids, labels[0].astype(int).tolist()))
