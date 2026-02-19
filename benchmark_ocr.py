#!/usr/bin/env python3
"""
Quick benchmark to compare OCR engine speeds.
"""

import subprocess
import time
from pathlib import Path

# Test image
TEST_IMAGE = Path("./floor5.jpg")
ENGINES = ["paddle", "tesseract", "easyocr"]

if not TEST_IMAGE.exists():
    print(f"Error: {TEST_IMAGE} not found")
    exit(1)

print("=" * 70)
print("OCR ENGINE SPEED BENCHMARK")
print("=" * 70)
print(f"Test image: {TEST_IMAGE.name}")
print()

results = {}

for engine in ENGINES:
    print(f"Testing {engine:15}...", end=" ", flush=True)

    cmd = [
        "python3", "extract_floor_plan.py",
        "--image", str(TEST_IMAGE),
        "--json-out", f"/tmp/bench_{engine}.json",
        "--svg-out", f"/tmp/bench_{engine}.svg",
        "--tessdata-dir", "./tessdata",
        "--ocr-engine", engine,
    ]

    try:
        start = time.time()
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        elapsed = time.time() - start

        # Extract room count
        rooms = 0
        for line in result.stdout.split('\n'):
            if line.startswith("Rooms:"):
                rooms = int(line.split(": ")[1].split(",")[0])

        results[engine] = {"time": elapsed, "rooms": rooms, "success": True}
        print(f"{elapsed:6.2f}s - {rooms} rooms found")
    except subprocess.TimeoutExpired:
        results[engine] = {"time": 120, "rooms": 0, "success": False}
        print(f"TIMEOUT (>120s)")
    except Exception as e:
        results[engine] = {"time": 0, "rooms": 0, "success": False}
        print(f"ERROR: {e}")

print()
print("=" * 70)
print("RESULTS:")
print("=" * 70)

# Sort by speed
sorted_results = sorted(results.items(), key=lambda x: x[1]["time"])

for engine, result in sorted_results:
    if result["success"]:
        speedup = results["paddle"]["time"] / result["time"]
        print(f"{engine:15} {result['time']:6.2f}s  {result['rooms']} rooms  ({speedup:.1f}x vs Paddle)")
    else:
        print(f"{engine:15} FAILED")

print()
print("RECOMMENDATION:")
best_engine = sorted_results[0][0]
best_time = sorted_results[0][1]["time"]
print(f"Use '{best_engine}' for fastest processing (~{best_time:.1f}s per image)")
print("=" * 70)
