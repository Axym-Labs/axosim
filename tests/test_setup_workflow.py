from pathlib import Path

from axosim.setup_workflow import DatasetSpec, SetupOptions, build_setup_plan


def test_setup_plan_uses_safe_default_without_large_downloads():
    plan = build_setup_plan(SetupOptions(project_root=Path("/repo"), data_dir=Path("/repo/data")))

    command_text = "\n".join(step.command for step in plan.steps if step.command)

    assert "pip install -e" in command_text
    assert "kaggle datasets download" not in command_text
    assert plan.next_steps


def test_setup_plan_adds_kaggle_downloads_only_when_requested():
    plan = build_setup_plan(
        SetupOptions(
            project_root=Path("/repo"),
            data_dir=Path("/repo/data"),
            install_kaggle=True,
            download_data=True,
        )
    )

    command_text = "\n".join(step.command for step in plan.steps if step.command)

    assert "pip install requests" in command_text
    assert "axosim.kaggle_download" in command_text
    assert "selfishgene/single-neurons-as-deep-nets-nmda-train-data" in command_text
    assert "selfishgene/single-neurons-as-deep-nets-nmda-test-data" in command_text


def test_setup_plan_can_include_conversion_command_for_raw_dir():
    plan = build_setup_plan(
        SetupOptions(
            project_root=Path("/repo"),
            data_dir=Path("/repo/data"),
            convert_raw=True,
            raw_dir=Path("/repo/data/raw/nmda-train"),
            shard_dir=Path("/repo/data/shards/nmda-train"),
            shard_size=64,
            datasets=(DatasetSpec("custom", "owner/dataset"),),
        )
    )

    command_text = "\n".join(step.command for step in plan.steps if step.command)

    assert "convert-neuronio" in command_text
    assert "--shard-size 64" in command_text
