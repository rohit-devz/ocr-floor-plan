#!/usr/bin/env python3
"""
Comprehensive test suite for all floor plan extraction methods.
Tests local OCR, Google Vision, and Gemini with performance comparison.

Usage:
    python3 test_all_extractors.py [--method local|google|gemini|all] [--quick]

Requirements:
    - Local OCR: just needs extract_floor_plan.py
    - Google Vision: GOOGLE_APPLICATION_CREDENTIALS env var set
    - Gemini: GEMINI_API_KEY env var set
"""

import subprocess
import json
import sys
import time
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Configuration
IMAGES_DIR = Path("./floor")
OUTPUT_DIR = Path("./test_results_comprehensive")
TESSDATA_DIR = "./tessdata"
OCR_ENGINE = "paddle"

# Extract methods
METHODS = {
    "local": {
        "name": "Local OCR (PaddleOCR)",
        "script": "extract_floor_plan.py",
        "timeout": 120,
        "requires_api_key": False,
        "desc": "OpenCV + PaddleOCR (free, ~30-40s/image)"
    },
    "google": {
        "name": "Google Cloud Vision",
        "script": "extract_floor_plan_google_vision.py",
        "timeout": 30,
        "requires_api_key": True,
        "env_var": "GOOGLE_APPLICATION_CREDENTIALS",
        "desc": "Google Cloud Vision API (1-3s/image, $0.0006/image)"
    },
    "gemini": {
        "name": "Gemini 2.5 Flash",
        "script": "extract_floor_plan_gemini.py",
        "timeout": 30,
        "requires_api_key": True,
        "env_var": "GEMINI_API_KEY",
        "desc": "Google Gemini Flash vision model (<1s/image, $0.000075/image)"
    }
}

def check_method_available(method: str) -> bool:
    """Check if a method is available (script exists, API key if needed)."""
    config = METHODS[method]
    script = Path(config["script"])

    if not script.exists():
        return False

    if config["requires_api_key"]:
        env_var = config.get("env_var")
        if env_var and not os.getenv(env_var):
            return False

    return True

def run_extraction(image_path: Path, method: str) -> dict:
    """Run floor plan extraction using specified method."""
    config = METHODS[method]

    output_base = OUTPUT_DIR / method / image_path.stem
    output_base.parent.mkdir(parents=True, exist_ok=True)

    output_json = output_base.with_suffix(".json")
    output_svg = output_base.with_suffix(".svg")

    cmd = ["python3", config["script"]]
    cmd.extend(["--image", str(image_path)])
    cmd.extend(["--json-out", str(output_json)])
    cmd.extend(["--svg-out", str(output_svg)])

    # Method-specific arguments
    if method == "local":
        cmd.extend(["--tessdata-dir", TESSDATA_DIR])
        cmd.extend(["--ocr-engine", OCR_ENGINE])
    elif method == "google":
        if os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
            cmd.extend(["--credentials", os.getenv("GOOGLE_APPLICATION_CREDENTIALS")])

    try:
        start_time = time.time()
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=config["timeout"]
        )
        elapsed_time = time.time() - start_time

        # Parse output for metrics
        room_count = 0
        room_names = []
        objects_count = 0
        scale = 0.0

        for line in result.stdout.split('\n'):
            if "Rooms:" in line and "Scale:" in line:
                # Format: Rooms: 5, Inner Walls: 8, Openings: 3, Objects: 2, Scale: 0.2119 mm/px
                parts = line.split(", ")
                try:
                    room_count = int(parts[0].split(": ")[1])
                    if len(parts) >= 4:
                        objects_count = int(parts[3].split(": ")[1])
                    if len(parts) >= 5:
                        scale = float(parts[4].split(": ")[1].split(" ")[0])
                except:
                    pass

        # Load JSON to get room names
        if output_json.exists():
            try:
                with open(output_json) as f:
                    data = json.load(f)
                    room_names = [r.get("name", "?") for r in data.get("rooms", [])]
                    room_count = len(room_names)
            except:
                pass

        return {
            "status": "success",
            "image": str(image_path),
            "method": method,
            "rooms": room_count,
            "room_names": room_names,
            "objects": objects_count,
            "scale": scale,
            "elapsed_time": elapsed_time,
            "json_output": str(output_json),
            "svg_output": str(output_svg),
            "error": None
        }

    except subprocess.TimeoutExpired:
        return {
            "status": "timeout",
            "image": str(image_path),
            "method": method,
            "error": f"Processing took too long (>{config['timeout']}s)",
            "elapsed_time": config["timeout"]
        }
    except Exception as e:
        return {
            "status": "error",
            "image": str(image_path),
            "method": method,
            "error": str(e),
            "elapsed_time": 0
        }

def main():
    """Main test runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Test all floor plan extraction methods")
    parser.add_argument("--method", choices=["local", "google", "gemini", "all"],
                       default="all", help="Which method to test")
    parser.add_argument("--quick", action="store_true", help="Only test first image")
    parser.add_argument("--skip-unavailable", action="store_true",
                       help="Skip methods with missing dependencies")
    args = parser.parse_args()

    print("=" * 100)
    print("COMPREHENSIVE FLOOR PLAN EXTRACTION TEST SUITE")
    print("=" * 100)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Determine which methods to test
    if args.method == "all":
        methods_to_test = list(METHODS.keys())
    else:
        methods_to_test = [args.method]

    # Check availability
    print("Method Availability:")
    print("-" * 100)
    available_methods = []
    for method in methods_to_test:
        config = METHODS[method]
        available = check_method_available(method)
        status = "✓" if available else "✗"
        reason = ""

        if not available:
            if not Path(config["script"]).exists():
                reason = f"(Script not found: {config['script']})"
            elif config["requires_api_key"]:
                env_var = config.get("env_var")
                reason = f"({env_var} not set)"
        else:
            available_methods.append(method)

        print(f"  {status} {config['name']:30} {reason}")

    print()

    if not available_methods:
        print("ERROR: No available extraction methods!")
        return 1

    if args.skip_unavailable:
        methods_to_test = available_methods

    # Find images
    if IMAGES_DIR.exists():
        image_files = sorted(IMAGES_DIR.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    else:
        image_files = []

    cwd_images = [
        Path("./floor.png"),
        Path("./floor2.png"),
        Path("./floor3.png"),
        Path("./floor4.jpg"),
        Path("./floor5.jpg"),
    ]
    cwd_images = [f for f in cwd_images if f.exists()]

    all_images = image_files + cwd_images
    all_images = list(dict.fromkeys(all_images))

    if args.quick and all_images:
        all_images = all_images[:1]

    if not all_images:
        print("ERROR: No images found!")
        return 1

    print(f"Found {len(all_images)} image(s):")
    for img in all_images:
        print(f"  - {img.name}")
    print()

    # Run tests
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}

    for method in methods_to_test:
        if method not in available_methods:
            continue

        print("=" * 100)
        print(f"Testing: {METHODS[method]['name']}")
        print(f"Description: {METHODS[method]['desc']}")
        print("=" * 100)

        config = METHODS[method]
        method_start = time.time()
        method_results = []

        for i, image_path in enumerate(all_images, 1):
            print(f"  [{i}/{len(all_images)}] {image_path.name}...", end=" ", flush=True)

            result = run_extraction(image_path, method)
            method_results.append(result)

            if result["status"] == "success":
                elapsed = result["elapsed_time"]
                rooms = result["rooms"]
                print(f"✓ {rooms} rooms in {elapsed:.2f}s")
            else:
                print(f"✗ {result['status'].upper()}: {result['error']}")

        method_elapsed = time.time() - method_start
        all_results[method] = {
            "config": config,
            "results": method_results,
            "total_time": method_elapsed
        }

        # Method summary
        successful = [r for r in method_results if r["status"] == "success"]
        if successful:
            total_rooms = sum(r["rooms"] for r in successful)
            total_time = sum(r["elapsed_time"] for r in successful)
            avg_time = total_time / len(successful)

            print()
            print(f"  Summary:")
            print(f"    Successful: {len(successful)}/{len(all_images)}")
            print(f"    Total rooms: {total_rooms}")
            print(f"    Average time: {avg_time:.2f}s per image")
            print()

    # Cross-method comparison
    if len(methods_to_test) > 1:
        print("=" * 100)
        print("CROSS-METHOD COMPARISON")
        print("=" * 100)
        print()

        # Find common successful images
        success_by_method = {}
        for method, data in all_results.items():
            success_by_method[method] = [r for r in data["results"] if r["status"] == "success"]

        common_images = set()
        for results in success_by_method.values():
            images = {Path(r["image"]).name for r in results}
            if not common_images:
                common_images = images
            else:
                common_images &= images

        if common_images:
            print("Performance Comparison (images processed by all methods):")
            print("-" * 100)

            # Header
            method_cols = []
            for method in methods_to_test:
                if method in all_results:
                    method_cols.append(method)

            print(f"{'Image':<30}", end="")
            for method in method_cols:
                print(f"{METHODS[method]['name']:<20}", end="")
            print()

            print("-" * 100)

            # Rows for each image
            for image_name in sorted(common_images):
                print(f"{image_name:<30}", end="")
                for method in method_cols:
                    results = [r for r in all_results[method]["results"]
                              if Path(r["image"]).name == image_name and r["status"] == "success"]
                    if results:
                        time_str = f"{results[0]['elapsed_time']:.2f}s"
                        print(f"{time_str:<20}", end="")
                    else:
                        print(f"{'FAILED':<20}", end="")
                print()

            print()
            print("Performance Summary:")
            print("-" * 100)

            for method in method_cols:
                results = all_results[method]["results"]
                successful = [r for r in results if r["status"] == "success"]
                if successful:
                    total_time = sum(r["elapsed_time"] for r in successful)
                    avg_time = total_time / len(successful)

                    print(f"{METHODS[method]['name']:<30}")
                    print(f"  Total time: {total_time:.2f}s")
                    print(f"  Average: {avg_time:.2f}s/image")

                    # Estimate for 100 images
                    est_100 = avg_time * 100 / 60
                    print(f"  Est. for 100 images: {est_100:.1f} minutes")
                    print()

    # Detailed results
    print("=" * 100)
    print("DETAILED RESULTS")
    print("=" * 100)

    for method in methods_to_test:
        if method not in all_results:
            continue

        print(f"\n{METHODS[method]['name']}:")
        print("-" * 100)

        results = all_results[method]["results"]
        for r in results:
            if r["status"] == "success":
                rooms_info = f"{r['rooms']} rooms"
                if r["room_names"]:
                    rooms_info += f" ({', '.join(r['room_names'])})"
                time_info = f"{r['elapsed_time']:.2f}s"
                print(f"  ✓ {Path(r['image']).name:40} {rooms_info:50} {time_info}")
            else:
                print(f"  ✗ {Path(r['image']).name:40} {r['status'].upper()}: {r['error']}")

    # Final summary
    print()
    print("=" * 100)
    print("TEST COMPLETE")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    return 0

if __name__ == "__main__":
    sys.exit(main())
