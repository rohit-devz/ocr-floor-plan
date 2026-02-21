import html
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
REPORT_FILE = "ocr_test_report.html"


def write_html_report(report_path: Path, floor_dir: Path, entries: list[dict]) -> None:
    rows: list[str] = []
    for index, entry in enumerate(entries, start=1):
        image_path = entry["image_path"]
        image_href = image_path.resolve().as_uri()
        output = f'{entry["stdout"]}{entry["stderr"]}'
        if entry["returncode"] != 0:
            output += f"\n[exit code: {entry['returncode']}]"
        output += f"\n[time: {entry['elapsed']:.3f}s]"
        rows.append(
            (
                "<tr>"
                f"<td>{index}</td>"
                f"<td>{html.escape(entry['completed_at'])}</td>"
                f"<td><a href=\"{html.escape(image_href)}\">{html.escape(str(image_path))}</a></td>"
                f"<td><code>{html.escape(entry['command'])}</code></td>"
                f"<td>{entry['returncode']}</td>"
                f"<td>{entry['elapsed']:.3f}s</td>"
                f"<td><pre>{html.escape(output)}</pre></td>"
                "</tr>"
            )
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Test Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f3f3f3; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; }}
    code {{ white-space: nowrap; }}
  </style>
</head>
<body>
  <h1>OCR Test Report</h1>
  <p><strong>Source directory:</strong> {html.escape(str(floor_dir))}</p>
  <p><strong>Last updated:</strong> {html.escape(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))}</p>
  <p><strong>Completed tests:</strong> {len(entries)}</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Completed At</th>
        <th>Image</th>
        <th>Command</th>
        <th>Exit Code</th>
        <th>Duration</th>
        <th>Output</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    report_path.write_text(report_html, encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    floor_dir = (script_dir / ".." / "floor").resolve()
    floor_dir = Path("/home/rohit/Desktop/work/floor_images")
    report_path = script_dir / REPORT_FILE
    entries: list[dict] = []

    write_html_report(report_path, floor_dir, entries)
    print(f"HTML report: {report_path}")

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
        command = f"py ocr.py {image_path}"
        print(f"\n$ {command}")
        start = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "ocr.py", str(image_path)],
            cwd=script_dir,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - start

        if result.stdout:
            print(result.stdout, end="" if result.stdout.endswith("\n") else "\n")
        if result.stderr:
            print(result.stderr, end="" if result.stderr.endswith("\n") else "\n")
        if result.returncode != 0:
            print(f"[exit code: {result.returncode}]")
        print(f"[time: {elapsed:.3f}s]")

        entries.append(
            {
                "image_path": image_path,
                "command": command,
                "stdout": result.stdout or "",
                "stderr": result.stderr or "",
                "returncode": result.returncode,
                "elapsed": elapsed,
                "completed_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        write_html_report(report_path, floor_dir, entries)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
