"""Quickstart script for YOLO Dataset Studio CLI."""

from pathlib import Path

from subprocess import run


def prepare_workspace(base_path: Path = Path("datasets/demo")) -> None:
    """Create a placeholder workspace structure for first-run demos."""

    images_dir = base_path / "images"
    labels_dir = base_path / "labels"

    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    print(f"Prepared empty dataset workspace at: {base_path.resolve()}")


def launch_cli() -> None:
    """Invoke the YOLO Dataset Studio interactive CLI."""

    result = run(["yolo-dataset-studio"], check=False)
    if result.returncode != 0:
        print("CLI exited with a non-zero status. Review the console output above for details.")


if __name__ == "__main__":
    prepare_workspace()
    launch_cli()
