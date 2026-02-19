#!/usr/bin/env python3
"""
Test script to extract floor plans from all images in the floor directory.
Processes each image and reports results.
"""

import subprocess
import json
import sys
import time
from pathlib import Path
from datetime import datetime

# Configuration
IMAGES_DIR = Path("./floor")
OUTPUT_DIR = Path("./test_results")
SCRIPT = "extract_floor_plan.py"
OCR_ENGINE = "paddle"
TESSDATA_DIR = "./tessdata"

def run_extraction(image_path: Path) -> dict:
    """Run floor plan extraction on a single image."""
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
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60
        )
        elapsed_time = time.time() - start_time

        # Parse output for room count
        output_lines = result.stdout.split('\n')
        room_count = 0
        inner_walls = 0
        openings = 0
        objects_count = 0
        scale = 0.0

        for line in output_lines:
            if line.startswith("Rooms:"):
                parts = line.split(", ")
                room_count = int(parts[0].split(": ")[1])
                inner_walls = int(parts[1].split(": ")[1])
                openings = int(parts[2].split(": ")[1])
                objects_count = int(parts[3].split(": ")[1])
            elif line.startswith("Scale:"):
                scale = float(line.split(": ")[1].split(" ")[0])

        # Try to load JSON to get room names
        room_names = []
        if output_json.exists():
            try:
                with open(output_json) as f:
                    data = json.load(f)
                    room_names = [r.get("name", "?") for r in data.get("rooms", [])]
            except:
                pass

        return {
            "status": "success",
            "image": str(image_path),
            "rooms": room_count,
            "room_names": room_names,
            "inner_walls": inner_walls,
            "openings": openings,
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
            "error": "Processing took too long (>60s)"
        }
    except Exception as e:
        return {
            "status": "error",
            "image": str(image_path),
            "error": str(e)
        }

def main():
    """Main test runner."""
    print("=" * 80)
    print("FLOOR PLAN EXTRACTION TEST SUITE")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Find all images
    if IMAGES_DIR.exists():
        image_files = sorted(IMAGES_DIR.glob("*.*"))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".png", ".jpeg"]]
    else:
        print(f"WARNING: {IMAGES_DIR} not found")
        image_files = []

    # Also check current directory
    cwd_images = [
        Path("./floor.png"),
        Path("./floor2.png"),
        Path("./floor3.png"),
        Path("./floor4.jpg"),
        Path("./floor5.jpg"),
    ]
    cwd_images = [f for f in cwd_images if f.exists()]

    all_images = image_files + cwd_images
    all_images = list(dict.fromkeys(all_images))  # Remove duplicates while preserving order

    if not all_images:
        print("ERROR: No images found!")
        return 1

    print(f"Found {len(all_images)} image(s) to process:")
    print()

    results = []
    total_start_time = time.time()
    for i, image_path in enumerate(all_images, 1):
        print(f"[{i}/{len(all_images)}] Processing: {image_path}")
        result = run_extraction(image_path)
        results.append(result)

        if result["status"] == "success":
            print(f"  ✓ Rooms: {result['rooms']}")
            if result['room_names']:
                print(f"    Names: {', '.join(result['room_names'])}")
            print(f"    Walls: {result['inner_walls']}, Openings: {result['openings']}, Objects: {result['objects']}")
            print(f"    Scale: {result['scale']} mm/px")
            print(f"    Time: {result['elapsed_time']:.2f}s")
        else:
            print(f"  ✗ {result['status'].upper()}: {result['error']}")
        print()
    total_elapsed_time = time.time() - total_start_time

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] != "success"]

    print(f"Total images processed: {len(results)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print()

    if successful:
        print("SUCCESSFUL EXTRACTIONS:")
        print("-" * 80)
        total_rooms = sum(r["rooms"] for r in successful)
        total_walls = sum(r["inner_walls"] for r in successful)
        total_time = sum(r["elapsed_time"] for r in successful)
        avg_time = total_time / len(successful) if successful else 0

        for r in successful:
            status = "✓"
            rooms_str = f"Rooms: {r['rooms']}"
            if r['room_names']:
                rooms_str += f" ({', '.join(r['room_names'])})"
            time_str = f"Time: {r['elapsed_time']:.2f}s"
            print(f"{status} {Path(r['image']).name:40} {rooms_str:50} {time_str}")

        print()
        print(f"Total rooms detected: {total_rooms}")
        print(f"Total inner walls detected: {total_walls}")
        print(f"Total extraction time: {total_time:.2f}s")
        print(f"Average time per image: {avg_time:.2f}s")

    if failed:
        print()
        print("FAILED EXTRACTIONS:")
        print("-" * 80)
        for r in failed:
            print(f"✗ {Path(r['image']).name:40} {r['status'].upper()}: {r['error']}")

    print()
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total test duration: {total_elapsed_time:.2f}s")
    print("=" * 80)

    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())
