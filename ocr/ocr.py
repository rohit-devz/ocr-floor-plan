import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from extract import extract_best_text, image_to_text  # slower

# from test2 import image_to_text  # faster
from utils import organize_floor_data


def run_both_ocr(image_path: str) -> dict:
    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {image_path}")

    with ThreadPoolExecutor(max_workers=2) as executor:
        future_fast = executor.submit(extract_best_text, str(path))
        future_simple = executor.submit(image_to_text, str(path))

        # Wait for both results, then return together.
        return {
            "test_py_result": future_fast.result(),
            "test2_py_result": future_simple.result(),
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run OCR from test.py and test2.py in parallel and return both outputs"
    )
    parser.add_argument("image_path", help="Input image path")
    args = parser.parse_args()

    results = run_both_ocr(args.image_path)
    # print("\n--- test.py (extract_best_text) ---\n")
    # print(results["test_py_result"] or "[No text found]")
    # print("\n--- test2.py (image_to_text) ---\n")
    # print(results["test2_py_result"] or "[No text found]")
    # result = organize_floor_data(results["test_py_result"], results["test2_py_result"])

    # print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
