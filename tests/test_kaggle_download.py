import json
import os
import zipfile

from bnn_sim.kaggle_download import (
    KaggleCredentials,
    build_dataset_url,
    find_zip_prefix_bytes,
    load_kaggle_credentials,
    parse_content_range_start,
    replace_with_retry,
    safe_extract_zip,
    usable_zip_exists,
    zip_members_present,
)


def test_load_kaggle_credentials_from_json(tmp_path):
    token_path = tmp_path / "kaggle.json"
    token_path.write_text(json.dumps({"username": "user", "key": "secret"}), encoding="utf-8")

    credentials = load_kaggle_credentials(token_path=token_path, env={})

    assert credentials == KaggleCredentials(username="user", key="secret")


def test_load_kaggle_credentials_prefers_environment():
    credentials = load_kaggle_credentials(
        token_path=None,
        env={"KAGGLE_USERNAME": "env-user", "KAGGLE_KEY": "env-key"},
    )

    assert credentials == KaggleCredentials(username="env-user", key="env-key")


def test_build_dataset_url_uses_download_endpoint():
    assert build_dataset_url("owner/dataset") == "https://www.kaggle.com/api/v1/datasets/download/owner/dataset"


def test_parse_content_range_start_extracts_start_byte():
    assert parse_content_range_start("bytes 1024-2047/4096") == 1024
    assert parse_content_range_start(None) is None


def test_safe_extract_zip_repairs_concatenated_prefix(tmp_path):
    clean_zip = tmp_path / "clean.zip"
    prefixed_zip = tmp_path / "prefixed.zip"
    extract_dir = tmp_path / "extract"

    with zipfile.ZipFile(clean_zip, "w") as archive:
        archive.writestr("folder/example.txt", "ok")
    prefixed_zip.write_bytes(b"stale-partial-prefix" + clean_zip.read_bytes())

    assert find_zip_prefix_bytes(prefixed_zip) == len(b"stale-partial-prefix")

    safe_extract_zip(prefixed_zip, extract_dir)

    assert (extract_dir / "folder" / "example.txt").read_text(encoding="utf-8") == "ok"
    assert find_zip_prefix_bytes(prefixed_zip) == 0


def test_safe_extract_zip_replaces_stale_repair_file(tmp_path):
    clean_zip = tmp_path / "clean.zip"
    prefixed_zip = tmp_path / "prefixed.zip"
    stale_repair = tmp_path / "prefixed.zip.repaired"
    extract_dir = tmp_path / "extract"

    with zipfile.ZipFile(clean_zip, "w") as archive:
        archive.writestr("example.txt", "ok")
    prefixed_zip.write_bytes(b"stale-prefix" + clean_zip.read_bytes())
    stale_repair.write_bytes(b"incomplete previous repair")

    safe_extract_zip(prefixed_zip, extract_dir)

    assert (extract_dir / "example.txt").read_text(encoding="utf-8") == "ok"
    assert not stale_repair.exists()


def test_usable_zip_exists_accepts_readable_zip(tmp_path):
    clean_zip = tmp_path / "clean.zip"

    with zipfile.ZipFile(clean_zip, "w") as archive:
        archive.writestr("example.txt", "ok")

    assert usable_zip_exists(clean_zip)
    assert not usable_zip_exists(tmp_path / "missing.zip")


def test_zip_members_present_detects_existing_extraction(tmp_path):
    clean_zip = tmp_path / "clean.zip"
    extract_dir = tmp_path / "extract"

    with zipfile.ZipFile(clean_zip, "w") as archive:
        archive.writestr("example.txt", "ok")

    assert not zip_members_present(clean_zip, extract_dir)
    safe_extract_zip(clean_zip, extract_dir)
    assert zip_members_present(clean_zip, extract_dir)


def test_replace_with_retry_handles_transient_permission_error(tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    target = tmp_path / "target.txt"
    source.write_text("ok", encoding="utf-8")
    calls = {"count": 0}
    original_replace = type(source).replace

    def flaky_replace(self, other):
        if self == source and calls["count"] == 0:
            calls["count"] += 1
            raise PermissionError(32, "locked", str(self))
        return original_replace(self, other)

    monkeypatch.setattr(type(source), "replace", flaky_replace)

    replace_with_retry(source, target, sleep=lambda _: None)

    assert calls["count"] == 1
    assert target.read_text(encoding="utf-8") == "ok"
    assert not source.exists()
