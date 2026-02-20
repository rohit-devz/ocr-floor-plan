import subprocess
import sys
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    floor_dir = (script_dir / ".." / "floor").resolve()

    if not floor_dir.is_dir():
        print(f"Floor directory not found: {floor_dir}")
        return 1

    images = sorted(
        [p for p in floor_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )

    if not images:
        print(f"No images found in: {floor_dir}")
        return 1

    for image_path in images:
        rel = Path("..") / "floor" / image_path.name
        print(f"\n$ py ocr.py {rel}")
        result = subprocess.run(
            [sys.executable, "ocr.py", str(rel)],
            cwd=script_dir,
            text=True,
            capture_output=True,
        )

        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if result.returncode != 0:
            print(f"[exit code: {result.returncode}]")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
