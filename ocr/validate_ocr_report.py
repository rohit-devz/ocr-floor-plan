import argparse
import html
import json
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_INPUT_JSON = "ocr_test_report.json"
DEFAULT_OUTPUT_JSON = "ocr_validation_report.json"
DEFAULT_OUTPUT_HTML = "ocr_validation_report.html"
TIME_LINE_RE = re.compile(r"\n?\[time:\s*\d+(?:\.\d+)?s\]\s*$")
ROOM_RE = re.compile(r"\b(KITCHEN|BEDROOM|LIVING|DINING|BATH|BALCONY|TOILET|LAUNDRY|CLOSET)\b")


def normalize_output(text: str) -> str:
    text = TIME_LINE_RE.sub("", text.strip())
    lines = [line.rstrip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def extract_signal_lines(text: str) -> list[str]:
    signal: list[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("*"):
            continue
        if line.startswith("[time:"):
            continue
        signal.append(line)
    return signal


def mismatch_reason(expected: str, actual: str) -> str:
    expected_lines = extract_signal_lines(expected)
    actual_lines = extract_signal_lines(actual)
    missing = [line for line in expected_lines if line not in actual_lines]
    extra = [line for line in actual_lines if line not in expected_lines]

    parts: list[str] = []
    if missing:
        parts.append(f"missing expected OCR lines: {missing}")
    if extra:
        parts.append(f"unexpected OCR lines: {extra}")
    if not parts:
        parts.append("output text differs after normalization")
    return "; ".join(parts)


def quality_failure_reason(text: str) -> str | None:
    lines = extract_signal_lines(text)
    if not lines:
        return "no OCR content extracted (only markers/empty output)"

    has_room = any(ROOM_RE.search(line.upper()) for line in lines)
    has_dimensions = any("{'width':" in line and "'height':" in line for line in lines)

    if not has_room:
        return "no recognizable room labels extracted"
    if not has_dimensions:
        none_dims = [line for line in lines if ROOM_RE.search(line.upper()) and line.endswith("None")]
        if none_dims:
            return "room labels found but dimensions are missing/None"
        return "no structured room dimensions extracted"
    return None


def resolve_command(command: str, python_bin: str, image_path: Path) -> list[str]:
    trimmed = command.strip()
    if trimmed.startswith("py "):
        return [python_bin, *trimmed.split()[1:]]
    if trimmed.startswith("python "):
        return [python_bin, *trimmed.split()[1:]]
    if trimmed.startswith("python3 "):
        return [python_bin, *trimmed.split()[1:]]
    if trimmed:
        return ["zsh", "-lc", trimmed]
    return [python_bin, "ocr.py", str(image_path)]


def run_scan(
    script_dir: Path, command: str, python_bin: str, image_path: Path
) -> tuple[int, str, float]:
    started = time.perf_counter()
    run_cmd = resolve_command(command, python_bin, image_path)
    result = subprocess.run(
        run_cmd,
        cwd=script_dir,
        text=True,
        capture_output=True,
    )
    elapsed = time.perf_counter() - started
    combined = f"{result.stdout or ''}{result.stderr or ''}"
    return result.returncode, combined, elapsed


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_html(path: Path, payload: dict) -> None:
    rows: list[str] = []
    for item in payload["results"]:
        image_href = Path(item["image_path"]).resolve().as_uri()
        status = item["status"]
        cls = "pass" if status == "PASS" else "fail"
        rows.append(
            (
                "<tr>"
                f"<td>{item['index']}</td>"
                f"<td class=\"{cls}\">{status}</td>"
                f"<td>{html.escape(item['reason'])}</td>"
                f"<td><a href=\"{html.escape(image_href)}\">{html.escape(item['image_path'])}</a></td>"
                f"<td>{item['expected_exit_code']}</td>"
                f"<td>{item['actual_exit_code']}</td>"
                f"<td>{item['duration_seconds']:.3f}s</td>"
                f"<td><pre>{html.escape(item['expected_output'])}</pre></td>"
                f"<td><pre>{html.escape(item['actual_output'])}</pre></td>"
                "</tr>"
            )
        )

    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>OCR Validation Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 20px; }}
    table {{ border-collapse: collapse; width: 100%; }}
    th, td {{ border: 1px solid #ccc; padding: 8px; vertical-align: top; text-align: left; }}
    th {{ background: #f3f3f3; }}
    pre {{ margin: 0; white-space: pre-wrap; word-break: break-word; max-width: 420px; }}
    .pass {{ color: #0a7a22; font-weight: 700; }}
    .fail {{ color: #b00020; font-weight: 700; }}
  </style>
</head>
<body>
  <h1>OCR Validation Report</h1>
  <p><strong>Input report:</strong> {html.escape(payload['input_report'])}</p>
  <p><strong>Source directory:</strong> {html.escape(payload['source_directory'])}</p>
  <p><strong>Validated at:</strong> {html.escape(payload['validated_at'])}</p>
  <p><strong>Total:</strong> {payload['summary']['total']} | <strong>Pass:</strong> {payload['summary']['pass']} | <strong>Fail:</strong> {payload['summary']['fail']}</p>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>Status</th>
        <th>Reason</th>
        <th>Image</th>
        <th>Expected Exit</th>
        <th>Actual Exit</th>
        <th>Duration</th>
        <th>Expected Output</th>
        <th>Actual Output</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    path.write_text(doc, encoding="utf-8")


def validate(input_json: Path, output_json: Path, output_html: Path, python_bin: str) -> int:
    script_dir = Path(__file__).resolve().parent
    data = json.loads(input_json.read_text(encoding="utf-8"))
    results: list[dict] = []

    for entry in data.get("entries", []):
        image_path = Path(entry["image_path"])
        expected_output_raw = str(entry.get("output", ""))
        expected_output = normalize_output(expected_output_raw)
        expected_exit = int(entry.get("exit_code", 0))
        command = str(entry.get("command", f"{sys.executable} ocr.py {image_path}"))

        if not image_path.is_file():
            results.append(
                {
                    "index": int(entry.get("index", len(results) + 1)),
                    "image_path": str(image_path),
                    "status": "FAIL",
                    "reason": "image file not found",
                    "expected_exit_code": expected_exit,
                    "actual_exit_code": None,
                    "duration_seconds": 0.0,
                    "expected_output": expected_output,
                    "actual_output": "",
                }
            )
            continue

        actual_exit, actual_raw, elapsed = run_scan(script_dir, command, python_bin, image_path)
        actual_output = normalize_output(actual_raw)

        if actual_exit != expected_exit:
            status = "FAIL"
            reason = f"exit code mismatch: expected {expected_exit}, got {actual_exit}"
        elif actual_output == expected_output:
            quality_reason = quality_failure_reason(actual_output)
            if quality_reason is None:
                status = "PASS"
                reason = "output matched and quality checks passed"
            else:
                status = "FAIL"
                reason = f"output matched baseline but failed quality checks: {quality_reason}"
        else:
            status = "FAIL"
            reason = mismatch_reason(expected_output, actual_output)

        results.append(
            {
                "index": int(entry.get("index", len(results) + 1)),
                "image_path": str(image_path),
                "status": status,
                "reason": reason,
                "expected_exit_code": expected_exit,
                "actual_exit_code": actual_exit,
                "duration_seconds": round(elapsed, 3),
                "expected_output": expected_output,
                "actual_output": actual_output,
            }
        )

    pass_count = sum(1 for item in results if item["status"] == "PASS")
    payload = {
        "input_report": str(input_json.resolve()),
        "source_directory": str(data.get("source_directory", "")),
        "validated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "summary": {
            "total": len(results),
            "pass": pass_count,
            "fail": len(results) - pass_count,
        },
        "results": results,
    }
    write_json(output_json, payload)
    write_html(output_html, payload)

    print(f"Validation JSON: {output_json}")
    print(f"Validation HTML: {output_html}")
    print(
        "Summary: "
        f"{payload['summary']['pass']} passed, "
        f"{payload['summary']['fail']} failed, "
        f"{payload['summary']['total']} total"
    )
    return 0 if payload["summary"]["fail"] == 0 else 2


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Validate OCR outputs by re-scanning images from ocr_test_report.json"
    )
    parser.add_argument(
        "--input-json",
        default=str(script_dir / DEFAULT_INPUT_JSON),
        help="Path to the baseline JSON report generated by test_ocr.py",
    )
    parser.add_argument(
        "--output-json",
        default=str(script_dir / DEFAULT_OUTPUT_JSON),
        help="Path for the validation JSON output",
    )
    parser.add_argument(
        "--output-html",
        default=str(script_dir / DEFAULT_OUTPUT_HTML),
        help="Path for the validation HTML output",
    )
    parser.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter used to execute commands like 'py ocr.py ...'",
    )
    args = parser.parse_args()

    return validate(
        Path(args.input_json),
        Path(args.output_json),
        Path(args.output_html),
        args.python_bin,
    )


if __name__ == "__main__":
    raise SystemExit(main())
