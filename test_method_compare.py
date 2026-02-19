#!/usr/bin/env python3
"""
Quick method comparison test - Test all available methods on a single image
and show detailed performance/accuracy comparison.

Usage:
    python3 test_method_compare.py [--image path/to/image.jpg]
"""

import subprocess
import json
import sys
import time
import os
from pathlib import Path
from typing import Optional
import argparse

METHODS = {
    "local": {
        "name": "Local OCR (PaddleOCR)",
        "script": "extract_floor_plan.py",
        "speed": "30-40s",
        "cost": "Free",
        "accuracy": "Good",
        "setup": "5 min"
    },
    "google": {
        "name": "Google Cloud Vision",
        "script": "extract_floor_plan_google_vision.py",
        "speed": "1-3s",
        "cost": "$0.0006/img",
        "accuracy": "Excellent",
        "setup": "30 min"
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "script": "extract_floor_plan_gemini.py",
        "speed": "<1s",
        "cost": "$0.000075/img",
        "accuracy": "Excellent",
        "setup": "5 min"
    }
}

def find_test_image() -> Optional[Path]:
    """Find first available test image."""
    candidates = [
        Path("./floor5.jpg"),
        Path("./floor4.jpg"),
        Path("./floor3.png"),
        Path("./floor2.png"),
        Path("./floor.png"),
    ]

    for img in candidates:
        if img.exists():
            return img

    # Search in floor directory
    floor_dir = Path("./floor")
    if floor_dir.exists():
        for img in floor_dir.glob("*.jpg"):
            return img
        for img in floor_dir.glob("*.png"):
            return img

    return None

def check_method_available(method: str) -> tuple[bool, str]:
    """Check if method is available and return reason if not."""
    script = Path(METHODS[method]["script"])

    if not script.exists():
        return False, f"Script not found: {script.name}"

    if method == "google":
        if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            return False, "GOOGLE_APPLICATION_CREDENTIALS env var not set"
    elif method == "gemini":
        if not os.getenv("GEMINI_API_KEY"):
            return False, "GEMINI_API_KEY env var not set"

    return True, ""

def test_method(image_path: Path, method: str) -> dict:
    """Test a single method on the image."""
    script = METHODS[method]["script"]
    output_dir = Path(f"./test_results_compare/{method}")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_json = output_dir / f"{image_path.stem}.json"
    output_svg = output_dir / f"{image_path.stem}.svg"

    cmd = ["python3", script]
    cmd.extend(["--image", str(image_path)])
    cmd.extend(["--json-out", str(output_json)])
    cmd.extend(["--svg-out", str(output_svg)])

    if method == "local":
        cmd.extend(["--tessdata-dir", "./tessdata"])
        cmd.extend(["--ocr-engine", "paddle"])

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start

        rooms = 0
        room_names = []
        if output_json.exists():
            try:
                data = json.load(open(output_json))
                rooms = len(data.get("rooms", []))
                room_names = [r.get("name", "?") for r in data.get("rooms", [])]
            except:
                pass

        return {
            "method": method,
            "status": "success",
            "time": elapsed,
            "rooms": rooms,
            "room_names": room_names,
            "json": str(output_json),
            "svg": str(output_svg)
        }
    except subprocess.TimeoutExpired:
        return {
            "method": method,
            "status": "timeout",
            "time": 120,
            "error": "Timeout (>120s)"
        }
    except Exception as e:
        return {
            "method": method,
            "status": "error",
            "time": 0,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser(description="Compare all available extraction methods")
    parser.add_argument("--image", type=Path, help="Test image path")
    parser.add_argument("--method", choices=["local", "google", "gemini"],
                       help="Test only specific method")
    args = parser.parse_args()

    # Find test image
    if args.image:
        test_image = args.image
    else:
        test_image = find_test_image()

    if not test_image or not test_image.exists():
        print("ERROR: No test image found!")
        print("Provide --image path/to/image.jpg")
        return 1

    print("=" * 100)
    print("FLOOR PLAN EXTRACTION METHOD COMPARISON")
    print("=" * 100)
    print(f"\nTest image: {test_image.name} ({test_image.stat().st_size / 1024 / 1024:.1f} MB)")
    print()

    # Check availability
    print("Method Availability:")
    print("-" * 100)

    available_methods = []
    for method in ["local", "google", "gemini"]:
        available, reason = check_method_available(method)
        status = "✓" if available else "✗"

        info = METHODS[method]
        print(f"{status} {info['name']:30} Speed: {info['speed']:8} Cost: {info['cost']:15} Setup: {info['setup']}")

        if not available:
            print(f"  → {reason}")
        else:
            available_methods.append(method)

    print()

    if args.method:
        if args.method in available_methods:
            available_methods = [args.method]
        else:
            available, reason = check_method_available(args.method)
            if not available:
                print(f"ERROR: Method not available: {reason}")
                return 1

    if not available_methods:
        print("ERROR: No methods available!")
        return 1

    # Run tests
    print(f"Testing on: {test_image.name}")
    print("-" * 100)
    print()

    results = []
    for method in available_methods:
        print(f"Testing {METHODS[method]['name']}...", end=" ", flush=True)
        result = test_method(test_image, method)
        results.append(result)

        if result["status"] == "success":
            print(f"✓ {result['rooms']} rooms in {result['time']:.2f}s")
        else:
            print(f"✗ {result['status']}: {result.get('error', 'Unknown error')}")

    print()
    print("=" * 100)
    print("COMPARISON RESULTS")
    print("=" * 100)
    print()

    successful = [r for r in results if r["status"] == "success"]

    if successful:
        # Speed comparison
        print("Speed Ranking (faster is better):")
        print("-" * 100)
        sorted_by_speed = sorted(successful, key=lambda r: r["time"])
        for i, r in enumerate(sorted_by_speed, 1):
            speedup = successful[0]["time"] / r["time"] if r["time"] > 0 else 1
            if i == 1:
                print(f"{i}. {METHODS[r['method']]['name']:<30} {r['time']:.2f}s (baseline)")
            else:
                print(f"{i}. {METHODS[r['method']]['name']:<30} {r['time']:.2f}s ({speedup:.1f}x faster)")

        print()
        print("Accuracy (rooms detected):")
        print("-" * 100)
        for r in sorted(successful, key=lambda x: -x["rooms"]):
            names = ", ".join(r["room_names"]) if r["room_names"] else "?"
            print(f"  {METHODS[r['method']]['name']:<30} {r['rooms']} rooms: {names}")

        print()
        print("Cost Estimate for 100 Images:")
        print("-" * 100)
        for r in successful:
            method = r["method"]
            info = METHODS[method]
            time_for_100 = r["time"] * 100 / 60  # minutes

            cost_per_100 = {
                "local": "$0.00 (free)",
                "google": "$0.06",
                "gemini": "$0.01"
            }

            print(f"  {info['name']:<30} {time_for_100:6.1f} min   {cost_per_100.get(method, 'N/A')}")

    # Recommendation
    print()
    print("=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    print()

    if not successful:
        print("No methods succeeded. Check requirements and API keys.")
    else:
        fastest = min(successful, key=lambda r: r["time"])
        most_accurate = max(successful, key=lambda r: r["rooms"])

        print(f"Fastest method:  {METHODS[fastest['method']]['name']}")
        print(f"Most accurate:   {METHODS[most_accurate['method']]['name']}")
        print()

        if fastest["method"] == "gemini":
            print("✓ Use Gemini Flash for:")
            print("  • Production workflows")
            print("  • Large batch processing")
            print("  • When speed matters most")
            print()
        elif fastest["method"] == "google":
            print("✓ Use Google Vision for:")
            print("  • Fast processing with accuracy")
            print("  • Medium-scale batch jobs")
            print()
        else:
            print("✓ Use Local OCR for:")
            print("  • No API setup required")
            print("  • Privacy-sensitive data")
            print("  • Offline processing")
            print()

    # Show output files
    if successful:
        print("Output Files:")
        print("-" * 100)
        for r in successful:
            print(f"  {METHODS[r['method']]['name']}:")
            print(f"    JSON: {r['json']}")
            print(f"    SVG:  {r['svg']}")

    print()
    return 0 if successful else 1

if __name__ == "__main__":
    sys.exit(main())
