#!/usr/bin/env python3
"""
Ultra-fast floor plan extraction using Gemini 2.0/2.5 Flash vision model.
Requires: pip install google-generativeai

Setup:
1. Get API key from https://aistudio.google.com/app/apikey
2. Set GEMINI_API_KEY environment variable
"""

import json
import sys
import time
import base64
import os
from pathlib import Path
from typing import Any

try:
    import google.generativeai as genai
except ImportError:
    print("Error: google-generativeai not installed")
    print("Install with: pip install google-generativeai")
    sys.exit(1)

import cv2
import numpy as np

# Import utilities from main script
sys.path.insert(0, str(Path(__file__).parent))
try:
    from extract_floor_plan import (
        detect_lines, detect_outer_frame, detect_room_components,
        build_inner_walls, detect_object_edges, detect_colored_openings,
        build_wall_defs, px_to_mm, write_svg,
        DEFAULT_CEILING_MM, DEFAULT_MM_PER_PX
    )
except ImportError:
    pass

def extract_with_gemini(image_path: Path, model: str = "gemini-2.5-flash") -> dict:
    """
    Extract floor plan data using Gemini vision model.

    Args:
        image_path: Path to floor plan image
        model: Gemini model to use (gemini-2.5-flash or gemini-2.0-flash)

    Returns:
        Structured floor plan data
    """

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable not set")

    genai.configure(api_key=api_key)

    # Read image
    with open(image_path, "rb") as f:
        image_data = base64.standard_b64encode(f.read()).decode("utf-8")

    # Determine image type
    suffix = image_path.suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif"
    }.get(suffix, "image/jpeg")

    # Prompt for structured extraction
    prompt = """Analyze this floor plan image and extract the following information in JSON format:

{
  "rooms": [
    {
      "name": "ROOM_NAME",
      "dimensions_ft": {"length": 12.5, "width": 10.0},
      "location": "description of location on plan"
    }
  ],
  "dimensions": {
    "total_width_ft": 30,
    "total_height_ft": 40
  },
  "walls": [
    {
      "type": "interior_wall",
      "length_ft": 15,
      "orientation": "horizontal or vertical"
    }
  ],
  "features": [
    {
      "name": "FEATURE_NAME (door, window, etc)",
      "type": "door|window|opening",
      "location": "which wall"
    }
  ],
  "fixtures": [
    {
      "name": "sink|toilet|bathtub|stove|etc",
      "room": "room name"
    }
  ],
  "scale_hint": "estimated mm per pixel if visible"
}

Instructions:
1. Extract all room labels and dimensions visible on the plan
2. List all interior and exterior walls with approximate dimensions
3. Identify doors, windows, and other openings
4. List fixtures (plumbing, kitchen appliances, etc)
5. All dimensions should be in FEET and INCHES format converted to decimal feet
6. Return ONLY valid JSON, no other text
7. If a dimension is not clearly visible, use "null"
8. Be precise with room names (BEDROOM, KITCHEN, LIVING, BATHROOM, etc)"""

    # Call Gemini API
    model_obj = genai.GenerativeModel(model)

    image_content = {
        "mime_type": mime_type,
        "data": image_data
    }

    response = model_obj.generate_content([prompt, image_content])

    # Parse response
    response_text = response.text.strip()

    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0]

        extracted = json.loads(response_text)
    except json.JSONDecodeError as e:
        print(f"Warning: Could not parse Gemini response as JSON: {e}", file=sys.stderr)
        print(f"Response was: {response_text[:200]}", file=sys.stderr)
        extracted = {"rooms": [], "error": "Failed to parse response"}

    return extracted

def gemini_to_floorplan_state(
    image_path: Path,
    gemini_data: dict,
    model: str = "gemini-2.5-flash"
) -> dict:
    """
    Convert Gemini extraction to floorplan state format.
    """

    # Read image for dimensions
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Use default scale
    mm_per_px = DEFAULT_MM_PER_PX

    # Parse room dimensions from Gemini data
    rooms = []
    name_counter = {}

    for i, room_data in enumerate(gemini_data.get("rooms", []), 1):
        room_name = room_data.get("name", f"ROOM {i}").upper()

        # Normalize name
        if "BEDROOM" in room_name:
            base_name = "BEDROOM"
        elif "KITCHEN" in room_name:
            base_name = "KITCHEN"
        elif "LIVING" in room_name:
            base_name = "LIVING"
        elif "BATHROOM" in room_name or "BATH" in room_name:
            base_name = "BATHROOM"
        elif "DINING" in room_name or "DRG" in room_name:
            base_name = "DRG / DINING"
        else:
            base_name = room_name

        # Handle duplicates
        count = name_counter.get(base_name, 0) + 1
        name_counter[base_name] = count
        final_name = f"{base_name} {count}" if count > 1 else base_name

        # Convert dimensions
        dims = room_data.get("dimensions_ft", {})
        length = dims.get("length")
        width = dims.get("width")

        # Estimate footprint (rough positioning)
        if length and width:
            w_mm = int(length * 304.8)
            h_mm = int(width * 304.8)
            min_x = (i % 3) * w_mm
            min_y = (i // 3) * h_mm
            max_x = min_x + w_mm
            max_y = min_y + h_mm
        else:
            min_x = (i % 3) * 2000
            min_y = (i // 3) * 2000
            max_x = min_x + 2000
            max_y = min_y + 2000

        room = {
            "id": i,
            "name": final_name,
            "label_text": room_name,
            "footprint": {
                "min_x": max(0, min_x),
                "max_x": max_x,
                "min_y": max(0, min_y),
                "max_y": max_y,
            }
        }

        if length and width:
            room["dimensions_ft"] = {"length": round(length, 2), "width": round(width, 2)}

        rooms.append(room)

    # Get total dimensions
    dims_data = gemini_data.get("dimensions", {})
    total_width = dims_data.get("total_width_ft", 40)
    total_height = dims_data.get("total_height_ft", 30)

    room_len_mm = int(total_width * 304.8)
    room_wid_mm = int(total_height * 304.8)

    # Create state
    state = {
        "units": "mm",
        "space": {
            "name": "Floor Plan",
            "height": DEFAULT_CEILING_MM,
            "footprint": {
                "min_x": 0,
                "min_y": 0,
                "max_x": room_len_mm,
                "max_y": room_wid_mm,
            },
        },
        "exterior_walls": build_wall_defs(room_len_mm, room_wid_mm),
        "openings": [],
        "columns": [],
        "inner_walls": [],
        "beams": [],
        "services": {"points": []},
        "objects": [],
        "rooms": rooms,
        "text_annotations": [],
        "meta": {
            "scale_mm_per_px": mm_per_px,
            "ocr_engine": f"gemini-{model}",
            "detected_room_labels": len(rooms),
            "source": "gemini_vision",
            "total_width_ft": total_width,
            "total_height_ft": total_height,
        },
    }

    return state

def extract_floor_plan_fast(
    image_path: Path,
    json_out: Path,
    svg_out: Path,
    model: str = "gemini-2.5-flash"
) -> None:
    """
    Fast floor plan extraction using Gemini.

    Args:
        image_path: Floor plan image
        json_out: Output JSON file
        svg_out: Output SVG file
        model: Gemini model (gemini-2.5-flash or gemini-2.0-flash)
    """

    print(f"Extracting with {model}...", file=sys.stderr)
    start_time = time.time()

    # Extract with Gemini
    gemini_data = extract_with_gemini(image_path, model)

    # Convert to state format
    state = gemini_to_floorplan_state(image_path, gemini_data, model)

    # Write outputs
    json_out.write_text(json.dumps(state, indent=2))
    write_svg(state, svg_out)

    elapsed = time.time() - start_time

    # Summary
    rooms = state.get("rooms", [])
    print(f"Wrote {json_out}")
    print(f"Wrote {svg_out}")
    print(f"Rooms: {len(rooms)}")
    if rooms:
        room_names = [r.get("name") for r in rooms]
        print(f"Names: {', '.join(room_names)}")
    print(f"Time: {elapsed:.2f}s")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Fast floor plan extraction using Gemini 2.5/2.0 Flash"
    )
    parser.add_argument("--image", required=True, help="Input floor plan image")
    parser.add_argument("--json-out", default="floorplan_state.json", help="Output JSON")
    parser.add_argument("--svg-out", default="final.svg", help="Output SVG")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        choices=["gemini-2.5-flash", "gemini-2.0-flash"],
        help="Gemini model to use"
    )
    args = parser.parse_args()

    extract_floor_plan_fast(
        Path(args.image),
        Path(args.json_out),
        Path(args.svg_out),
        args.model
    )
