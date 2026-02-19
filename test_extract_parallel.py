#!/usr/bin/env python3
"""
Parallel batch processor for floor plan extraction.
Processes multiple images simultaneously using multiprocessing.
"""

import subprocess
import json
import sys
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count
from datetime import datetime

# Configuration
IMAGES_DIR = Path("./floor")
OUTPUT_DIR = Path("./test_results_parallel")
SCRIPT = "extract_floor_plan.py"
OCR_ENGINE = "paddle"
TESSDATA_DIR = "./tessdata"
MAX_WORKERS = min(cpu_count(), 4)  # Cap at 4 to avoid memory issues

def process_image(image_path: Path) -> dict:
    """Process a single image."""
    output_json = OUTPUT_DIR / f"{image_path.stem}_floorplan_state.json"
    output_svg = OUTPUT_DIR / f"{image_path.stem}_final.svg"

    cmd = [
        "python3", SCRIPT,
        "--image", str(image_path),
        "--json-out", str(output_json),
        "--svg-out", str(output_svg),
        "--tessdata-dir", TESSDATA_DIR,
        "--ocr-engine", OCR_ENGINE,
    ]

    try:
        start_time = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed_time = time.time() - start_time

        # Parse output
        room_count = 0
        for line in result.stdout.split('\n'):
            if line.startswith("Rooms:"):
                room_count = int(line.split(": ")[1].split(",")[0])

        room_names = []
        if output_json.exists():
            try:
                with open(output_json) as f:
                    data = json.load(f)
                    room_names = [r.get("name", "?") for r in data.get("rooms", [])]
            except:
                pass

        return {
            "image": str(image_path),
            "status": "success",
            "rooms": room_count,
            "room_names": room_names,
            "elapsed_time": elapsed_time,
        }
    except Exception as e:
        return {
            "image": str(image_path),
            "status": "error",
            "error": str(e),
            "elapsed_time": 0,
        }

def main():
    print("=" * 80)
    print("PARALLEL FLOOR PLAN EXTRACTION TEST")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Workers: {MAX_WORKERS} processes")
    print()

    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all images
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

    if not all_images:
        print("ERROR: No images found!")
        return 1

    print(f"Found {len(all_images)} image(s)")
    print()

    # Process in parallel
    print("Processing...")
    test_start_time = time.time()

    with Pool(MAX_WORKERS) as pool:
        results = pool.map(process_image, all_images)

    total_elapsed_time = time.time() - test_start_time

    # Summary
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    for r in results:
        if r["status"] == "success":
            names_str = f"({', '.join(r['room_names'])})" if r['room_names'] else ""
            print(f"✓ {Path(r['image']).name:40} {r['rooms']} rooms {names_str:50} {r['elapsed_time']:.1f}s")
        else:
            print(f"✗ {Path(r['image']).name:40} {r['error']}")

    print()
    print("-" * 80)
    print(f"Total images: {len(all_images)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        total_rooms = sum(r["rooms"] for r in successful)
        total_time = sum(r["elapsed_time"] for r in successful)
        avg_time = total_time / len(successful)

        print(f"Total rooms detected: {total_rooms}")
        print(f"Sequential time (sum): {total_time:.1f}s")
        print(f"Parallel time: {total_elapsed_time:.1f}s")
        print(f"Speedup: {total_time / total_elapsed_time:.1f}x ({MAX_WORKERS} workers)")
        print(f"Average per image: {avg_time:.1f}s (sequential)")

    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
