from __future__ import annotations

from dataclasses import dataclass
import argparse
import json
import os
from pathlib import Path
import shutil
import time
from typing import Mapping
import zipfile

import requests


@dataclass(frozen=True)
class KaggleCredentials:
    username: str
    key: str


def load_kaggle_credentials(
    *,
    token_path: str | Path | None = None,
    env: Mapping[str, str] | None = None,
) -> KaggleCredentials:
    env = os.environ if env is None else env
    username = env.get("KAGGLE_USERNAME")
    key = env.get("KAGGLE_KEY")
    if username and key:
        return KaggleCredentials(username=username, key=key)

    token_path = Path.home() / ".kaggle" / "kaggle.json" if token_path is None else Path(token_path)
    payload = json.loads(token_path.read_text(encoding="utf-8"))
    return KaggleCredentials(username=payload["username"], key=payload["key"])


def build_dataset_url(slug: str) -> str:
    return f"https://www.kaggle.com/api/v1/datasets/download/{slug}"


def download_dataset_zip(
    slug: str,
    output_path: str | Path,
    *,
    credentials: KaggleCredentials | None = None,
    chunk_size: int = 8 * 1024 * 1024,
    progress_interval: float = 10.0,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if usable_zip_exists(output_path):
        print(f"{output_path.name}: existing readable ZIP found; skipping download", flush=True)
        return output_path

    part_path = output_path.with_suffix(output_path.suffix + ".part")
    credentials = load_kaggle_credentials() if credentials is None else credentials
    existing_size = part_path.stat().st_size if part_path.exists() else 0
    headers = {"Range": f"bytes={existing_size}-"} if existing_size else {}

    with requests.get(
        build_dataset_url(slug),
        auth=(credentials.username, credentials.key),
        headers=headers,
        stream=True,
        timeout=60,
    ) as response:
        if response.status_code == 416 and part_path.exists():
            replace_with_retry(part_path, output_path)
            return output_path
        response.raise_for_status()
        if existing_size and response.status_code == 206:
            range_start = parse_content_range_start(response.headers.get("content-range"))
            if range_start != existing_size:
                existing_size = 0
                part_path.unlink(missing_ok=True)
                response.close()
                return download_dataset_zip(
                    slug,
                    output_path,
                    credentials=credentials,
                    chunk_size=chunk_size,
                    progress_interval=progress_interval,
                )
        elif existing_size:
            existing_size = 0
            part_path.unlink(missing_ok=True)

        mode = "ab" if existing_size else "wb"
        downloaded = existing_size
        total = _total_size(response, existing_size)
        last_progress = 0.0
        with part_path.open(mode + "") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_progress >= progress_interval:
                    _print_progress(output_path.name, downloaded, total)
                    last_progress = now

    replace_with_retry(part_path, output_path)
    _print_progress(output_path.name, output_path.stat().st_size, output_path.stat().st_size)
    return output_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Resumably download a Kaggle dataset ZIP.")
    parser.add_argument("slug")
    parser.add_argument("output")
    parser.add_argument("--unzip-to")
    args = parser.parse_args(argv)

    output_path = download_dataset_zip(args.slug, args.output)
    if args.unzip_to:
        unzip_dir = Path(args.unzip_to)
        unzip_dir.mkdir(parents=True, exist_ok=True)
        print(f"extracting {output_path} to {unzip_dir}", flush=True)
        safe_extract_zip(output_path, unzip_dir)
    return 0


def safe_extract_zip(zip_path: str | Path, extract_dir: str | Path) -> None:
    zip_path = Path(zip_path)
    extract_dir = Path(extract_dir)
    prefix_bytes = find_zip_prefix_bytes(zip_path)
    if prefix_bytes > 0:
        print(f"repairing {zip_path.name}: removing {prefix_bytes} stale prefix bytes", flush=True)
        trim_zip_prefix(zip_path, prefix_bytes)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            if zip_members_present(zip_path, extract_dir, archive=archive):
                print(f"{zip_path.name}: extracted members already present; skipping extraction", flush=True)
                return
            archive.extractall(extract_dir)
    except zipfile.BadZipFile as error:
        bad_path = zip_path.with_suffix(zip_path.suffix + ".bad")
        if bad_path.exists():
            bad_path.unlink()
        replace_with_retry(zip_path, bad_path)
        raise zipfile.BadZipFile(
            f"{zip_path} is not a valid ZIP archive; moved corrupt file to {bad_path}. "
            "Rerun the setup command to redownload."
        ) from error


def zip_members_present(
    zip_path: str | Path,
    extract_dir: str | Path,
    *,
    archive: zipfile.ZipFile | None = None,
) -> bool:
    extract_dir = Path(extract_dir)
    close_archive = archive is None
    archive = zipfile.ZipFile(zip_path) if archive is None else archive
    try:
        infos = [info for info in archive.infolist() if not info.is_dir()]
        if not infos:
            return False
        for info in infos:
            target = extract_dir / info.filename
            if not target.exists() or target.stat().st_size != info.file_size:
                return False
        return True
    finally:
        if close_archive:
            archive.close()


def usable_zip_exists(zip_path: str | Path) -> bool:
    zip_path = Path(zip_path)
    if not zip_path.exists():
        return False
    try:
        with zipfile.ZipFile(zip_path) as archive:
            return bool(archive.infolist())
    except zipfile.BadZipFile:
        return False


def find_zip_prefix_bytes(zip_path: str | Path) -> int:
    zip_path = Path(zip_path)
    with zipfile.ZipFile(zip_path) as archive:
        offsets = [info.header_offset for info in archive.infolist()]
    return min(offsets) if offsets else 0


def trim_zip_prefix(zip_path: str | Path, prefix_bytes: int) -> None:
    zip_path = Path(zip_path)
    repaired_path = zip_path.with_suffix(zip_path.suffix + ".repaired")
    repaired_path.unlink(missing_ok=True)
    with zip_path.open("rb") as source:
        source.seek(prefix_bytes)
        with repaired_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=16 * 1024 * 1024)
    replace_with_retry(repaired_path, zip_path)


def replace_with_retry(
    source: str | Path,
    target: str | Path,
    *,
    attempts: int = 10,
    delay_seconds: float = 1.0,
    sleep=time.sleep,
) -> None:
    source = Path(source)
    target = Path(target)
    last_error: PermissionError | None = None
    for _ in range(attempts):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            last_error = error
            sleep(delay_seconds)
    assert last_error is not None
    raise last_error


def parse_content_range_start(value: str | None) -> int | None:
    if not value:
        return None
    unit, _, range_and_total = value.partition(" ")
    if unit.lower() != "bytes":
        return None
    start_text, _, _ = range_and_total.partition("-")
    try:
        return int(start_text)
    except ValueError:
        return None


def _total_size(response: requests.Response, existing_size: int) -> int | None:
    content_range = response.headers.get("content-range")
    if content_range and "/" in content_range:
        return int(content_range.rsplit("/", 1)[1])
    content_length = response.headers.get("content-length")
    if content_length is None:
        return None
    return existing_size + int(content_length)


def _print_progress(name: str, downloaded: int, total: int | None) -> None:
    downloaded_gib = downloaded / (1024**3)
    if total:
        total_gib = total / (1024**3)
        percent = 100 * downloaded / total
        print(f"{name}: {downloaded_gib:.2f}/{total_gib:.2f} GiB ({percent:.1f}%)", flush=True)
    else:
        print(f"{name}: {downloaded_gib:.2f} GiB", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
