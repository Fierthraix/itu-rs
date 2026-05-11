from __future__ import annotations

import hashlib
import os
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path
from typing import Optional

DATA_URL = "https://github.com/Fierthraix/itu-rs/releases/download/itu-rs-data-v2/itu-rs-data-v2.zip"
DATA_SHA256 = "f818af57daeb839e42ca5d45cac2fc838f5abe9c3a9303609cf9d79497eba7d6"
DATA_VERSION = "itu-rs-data-v2"
SENTINEL = Path("1511/v2_lat.npz")


def ensure_data_dir() -> Path:
    configured = os.environ.get("ITU_RS_DATA_DIR")
    if configured:
        return Path(configured)

    local = _local_checkout_data()
    if local is not None:
        os.environ["ITU_RS_DATA_DIR"] = str(local)
        return local

    data_dir = _cache_root() / DATA_VERSION / "data"
    if not (data_dir / SENTINEL).exists():
        _install_cached_data(data_dir)

    os.environ["ITU_RS_DATA_DIR"] = str(data_dir)
    return data_dir


def _local_checkout_data() -> Optional[Path]:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "data"
        if (candidate / SENTINEL).exists():
            return candidate
    return None


def _cache_root() -> Path:
    configured = os.environ.get("ITU_RS_DATA_CACHE")
    if configured:
        return Path(configured)

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base:
            return Path(base) / "itu-rs"
    elif sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "itu-rs"

    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base) if base else Path.home() / ".cache") / "itu-rs"


def _install_cached_data(data_dir: Path) -> None:
    cache = data_dir.parent.parent
    archive = cache / f"{DATA_VERSION}.zip"
    cache.mkdir(parents=True, exist_ok=True)

    if not archive.exists():
        _download_archive(archive)
    _verify_archive(archive)

    extract_root = data_dir.parent
    tmp_root = extract_root.with_name(f"{extract_root.name}.tmp")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_data = tmp_root / "data"
    tmp_data.mkdir(parents=True, exist_ok=True)

    try:
        _extract_archive(archive, tmp_data)
        if not (tmp_data / SENTINEL).exists():
            raise RuntimeError(f"{archive} did not contain data/{SENTINEL}")
        if extract_root.exists():
            shutil.rmtree(extract_root)
        tmp_root.replace(extract_root)
    finally:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)


def _download_archive(archive: Path) -> None:
    tmp = archive.with_suffix(".zip.tmp")
    try:
        with urllib.request.urlopen(DATA_URL) as response, tmp.open("wb") as file:
            shutil.copyfileobj(response, file)
        tmp.replace(archive)
    finally:
        if tmp.exists():
            tmp.unlink()


def _verify_archive(archive: Path) -> None:
    digest = hashlib.sha256()
    with archive.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != DATA_SHA256:
        raise RuntimeError(f"SHA256 mismatch for {archive}: expected {DATA_SHA256}, got {actual}")


def _extract_archive(archive: Path, data_dir: Path) -> None:
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            rel_path = _data_relative_path(member.filename)
            if rel_path is None:
                continue
            target = data_dir / rel_path
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)


def _data_relative_path(name: str) -> Optional[Path]:
    parts = Path(name).parts
    try:
        data_idx = parts.index("data")
    except ValueError:
        return None

    rel = Path(*parts[data_idx + 1 :])
    if rel.is_absolute() or ".." in rel.parts or not rel.parts:
        return None
    return rel
