import html
import json
import re
import subprocess
import sys
import time
from pathlib import Path


DATA_FILE = "data_image.json"
REPORT_HTML_FILE = "ocr_data_test_report.html"
KITCHEN_DIM_PATTERN = re.compile(
    r"KITCHEN[^\n]*\{'width':\s*\{'feet':\s*(\d+),\s*'inches':\s*(\d+)\},\s*'height':\s*\{'feet':\s*(\d+),\s*'inches':\s*(\d+)\}\}",
    re.IGNORECASE,
)


def parse_expected(value):
    if value is None:
        return None
    return float(value)


def extract_kitchen_dims(output_text: str):
    matches = KITCHEN_DIM_PATTERN.findall(output_text or "")
    if not matches:
        return None, None
    w_ft, w_in, h_ft, h_in = matches[-1]
    length = float(f"{int(w_ft)}.{int(w_in)}")
    bredth = float(f"{int(h_ft)}.{int(h_in)}")
    return length, bredth


def is_match(expected_length, expected_bredth, actual_length, actual_bredth) -> bool:
    if expected_length is None and expected_bredth is None:
        return actual_length is None and actual_bredth is None
    if actual_length is None or actual_bredth is None:
        return False
    return expected_length == actual_length and expected_bredth == actual_bredth


def write_html_report(report_path: Path, entries: list[dict], total: int, passed: int, failed: int) -> None:
    rows = []
    for i, e in enumerate(entries, start=1):
        image_href = Path(e["image_path"]).resolve().as_uri()
        rows.append(
            "<tr>"
            f"<td>{i}</td>"
            f"<td><a href=\"{html.escape(image_href)}\">{html.escape(e['image_path'])}</a></td>"
            f"<td>{html.escape(str(e['expected_length']))} x {html.escape(str(e['expected_bredth']))}</td>"
            f"<td>{html.escape(str(e['actual_length']))} x {html.escape(str(e['actual_bredth']))}</td>"
            f"<td>{e['returncode']}</td>"
            f"<td>{e['elapsed']:.2f}s</td>"
            f"<td>{e['status']}</td>"
            "</tr>"
        )

    report_html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Data Test Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; text-align: left; vertical-align: top; }}
    th {{ background: #f3f3f3; }}
    .PASS {{ color: #0a7a24; font-weight: bold; }}
    .FAIL {{ color: #b00020; font-weight: bold; }}
  </style>
</head>
<body>
  <h1>OCR Data Test Report</h1>
  <p><strong>Total:</strong> {total} | <strong>Pass:</strong> {passed} | <strong>Fail:</strong> {failed}</p>
  <table>
    <thead>
      <tr>
        <th>#</th><th>Image</th><th>Expected (L x B)</th><th>Actual (L x B)</th><th>Exit Code</th><th>Duration</th><th>Status</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>"""
    report_path.write_text(report_html, encoding="utf-8")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    data_path = script_dir / DATA_FILE
    report_path = script_dir / REPORT_HTML_FILE
    venv_python = script_dir.parent / ".venv" / "bin" / "python"
    python_cmd = str(venv_python) if venv_python.is_file() else sys.executable

    if not data_path.is_file():
        print(f"Missing data file: {data_path}")
        return 1

    rows = json.loads(data_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows:
        print("data_image.json is empty or not an array")
        return 1

    total = 0
    passed = 0
    entries = []

    for idx, row in enumerate(rows, start=1):
        image_path = row.get("image_path")
        expected_length = parse_expected(row.get("length"))
        expected_bredth = parse_expected(row.get("bredth"))

        if not image_path:
            print(f"[{idx}] FAIL image_path missing")
            total += 1
            continue

        start = time.perf_counter()
        result = subprocess.run(
            [python_cmd, "ocr.py", image_path],
            cwd=script_dir,
            text=True,
            capture_output=True,
        )
        elapsed = time.perf_counter() - start
        output = f"{result.stdout or ''}{result.stderr or ''}"

        actual_length, actual_bredth = extract_kitchen_dims(output)
        ok = result.returncode == 0 and is_match(
            expected_length, expected_bredth, actual_length, actual_bredth
        )

        status = "PASS" if ok else "FAIL"
        entries.append(
            {
                "image_path": image_path,
                "expected_length": expected_length,
                "expected_bredth": expected_bredth,
                "actual_length": actual_length,
                "actual_bredth": actual_bredth,
                "returncode": result.returncode,
                "elapsed": elapsed,
                "status": status,
            }
        )
        print(
            f"[{idx}] {status} expected=({expected_length}x{expected_bredth}) "
            f"actual=({actual_length}x{actual_bredth}) time={elapsed:.2f}s image={image_path}"
        )
        if not ok and result.returncode != 0:
            print(f"    exit_code={result.returncode}")

        total += 1
        if ok:
            passed += 1

    failed = total - passed
    write_html_report(report_path, entries, total, passed, failed)
    print(f"HTML report: {report_path}")
    print(f"\nSummary: total={total} pass={passed} fail={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
