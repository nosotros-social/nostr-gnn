import shlex
import subprocess
from pathlib import Path

import modal

APP_NAME = "nostr-gnn"
PROJECT_ROOT = Path("/app")
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_VOLUME_NAME = "nostr-gnn-data"
OUTPUT_VOLUME_NAME = "nostr-gnn-outputs"
GPU = "L4"

app = modal.App(APP_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_pyproject("pyproject-modal.toml")
    .pip_install(
        "pyg-lib",
        "torch-sparse",
        find_links="https://data.pyg.org/whl/torch-2.10.0+cu128.html",
    )
    .add_local_dir("src", remote_path=str(PROJECT_ROOT / "src"))
    .add_local_file(
        "config.modal.yaml",
        remote_path=str(PROJECT_ROOT / "config.modal.yaml"),
    )
    .add_local_file(
        "pyproject-modal.toml",
        remote_path=str(PROJECT_ROOT / "pyproject-modal.toml"),
    )
)

data_volume = modal.Volume.from_name(DATA_VOLUME_NAME, create_if_missing=True)
output_volume = modal.Volume.from_name(OUTPUT_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


def _override_args(overrides: str) -> list[str]:
    return shlex.split(overrides) if overrides.strip() else []


@app.function(
    image=image,
    secrets=[hf_secret],
    volumes={
        DATA_DIR: data_volume,
        OUTPUT_DIR: output_volume,
    },
    cpu=8,
    memory=32768,
    timeout=60 * 60 * 12,
    gpu=GPU,
)
def train(overrides: str = "") -> None:
    cmd = ["python", "src/run.py", *_override_args(overrides)]
    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    finally:
        output_volume.commit()


@app.local_entrypoint()
def main(
    action: str = "train", overrides: str = "", local_data_dir: str = "data"
) -> None:
    if action == "upload-data":
        local_path = Path(local_data_dir)
        if not local_path.exists():
            raise FileNotFoundError(f"Local data directory not found: {local_path}")
        for entry in data_volume.listdir("/"):
            data_volume.remove_file(entry.path, recursive=True)
        with data_volume.batch_upload(force=True) as batch:
            batch.put_directory(local_path, "/")
        print(f"Uploaded {local_path} to Modal volume '{DATA_VOLUME_NAME}'")
        return

    if action == "train":
        print(f"Starting Modal training with accelerator: {GPU}")
        train.remote(overrides=overrides)
        return

    raise ValueError(f"Unknown action: {action}")
