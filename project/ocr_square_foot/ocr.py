import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extract import extract_best_text, image_to_text
from utils import organize_floor_data


def run_both_ocr(image_path: str) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_fast = executor.submit(extract_best_text, str(path))
        future_simple = executor.submit(image_to_text, str(path))
        fast_text = future_fast.result()
        simple_text = future_simple.result()

    structured = organize_floor_data(fast_text, simple_text)
    return {
        "image_path": str(path.resolve()),
        "test_py_result": fast_text,
        "test2_py_result": simple_text,
        "structured": structured,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OCR parser with square-foot area support (e.g., 72 SQFT)"
    )
    parser.add_argument("image_path", help="Input image path")
    parser.add_argument(
        "--json-out",
        help="Optional output file path for structured JSON result",
    )
    args = parser.parse_args()

    result = run_both_ocr(args.image_path)
    rendered = json.dumps(result, indent=2)
    print(rendered)

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(rendered, encoding="utf-8")
        print(f"JSON written: {out_path}")


if __name__ == "__main__":
    main()
