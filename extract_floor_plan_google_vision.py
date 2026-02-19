#!/usr/bin/env python3
"""
Fast floor plan extraction using Google Cloud Vision API.
Requires: pip install google-cloud-vision

Setup:
1. Create GCP project and enable Vision API
2. Create service account and download JSON key
3. Set GOOGLE_APPLICATION_CREDENTIALS environment variable
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from google.cloud import vision
from google.oauth2 import service_account

# Import the core extraction functions from the main script
sys.path.insert(0, str(Path(__file__).parent))
from extract_floor_plan import (
    detect_lines, detect_outer_frame, detect_room_components,
    build_labeled_rooms_from_lines, merge_room_sets,
    clamp_rooms_to_space, filter_invalid_rooms, detect_object_edges,
    detect_colored_openings, contour_to_opening, build_wall_defs,
    assign_wall, px_to_mm, build_inner_walls, write_svg,
    contour_passes_opening_shape_filter, is_room_label_token,
    _extract_dimension_near_label, normalize_label_name,
    DEFAULT_CEILING_MM, DEFAULT_MM_PER_PX, ROOM_KEYWORDS
)

def ocr_tokens_google_vision(image_path: Path) -> list[dict]:
    """Use Google Cloud Vision for fast OCR."""
    try:
        client = vision.ImageAnnotatorClient()

        with open(image_path, 'rb') as f:
            content = f.read()

        image = vision.Image(content=content)
        response = client.document_text_detection(image=image)

        tokens = []
        img = cv2.imread(str(image_path))
        h, w = img.shape[:2]

        if response.text_annotations:
            # First annotation is full text, skip it
            for annotation in response.text_annotations[1:]:
                text = annotation.description.strip()
                if not text:
                    continue

                # Get bounding box
                vertices = annotation.bounding_poly.vertices
                if not vertices:
                    continue

                xs = [v.x for v in vertices]
                ys = [v.y for v in vertices]
                x0, x1 = int(min(xs)), int(max(xs))
                y0, y1 = int(min(ys)), int(max(ys))

                w_box = max(1, x1 - x0)
                h_box = max(1, y1 - y0)

                tokens.append({
                    "text": text,
                    "conf": 95.0,  # Google Vision is highly confident
                    "x": x0,
                    "y": y0,
                    "w": w_box,
                    "h": h_box,
                    "cx": x0 + w_box / 2.0,
                    "cy": y0 + h_box / 2.0,
                })

        return tokens

    except Exception as e:
        print(f"Google Vision error: {e}", file=sys.stderr)
        print("Falling back to local OCR...", file=sys.stderr)
        from extract_floor_plan import ocr_tokens
        img = cv2.imread(str(image_path))
        return ocr_tokens(img, engine="paddle")

def extract_floorplan_fast(image_path: Path) -> dict:
    """Fast extraction using Google Cloud Vision."""
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # Use Google Vision for OCR (much faster)
    tokens = ocr_tokens_google_vision(image_path)

    # Rest of processing is the same
    merged_lines = detect_lines(gray)
    left, right, top, bottom = detect_outer_frame(merged_lines, w, h)
    components = detect_room_components(gray)

    mm_per_px = DEFAULT_MM_PER_PX
    room_len_mm = max(1, px_to_mm(right - left, mm_per_px))
    room_wid_mm = max(1, px_to_mm(bottom - top, mm_per_px))

    rooms_comp = []  # Skip component-based (slower)
    rooms_line = build_labeled_rooms_from_lines(tokens, merged_lines, left, right, top, bottom, mm_per_px)
    rooms = rooms_line

    clamp_rooms_to_space(rooms, room_len_mm, room_wid_mm)
    rooms = filter_invalid_rooms(rooms)

    inner_walls = build_inner_walls(merged_lines, left, right, top, bottom, mm_per_px)
    objects = detect_object_edges(gray, left, right, top, bottom, mm_per_px, tokens)

    door_cnts, win_cnts = detect_colored_openings(img)
    openings = []
    for c in door_cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        wall_id = assign_wall(cx, cy, left, right, top, bottom)
        if not contour_passes_opening_shape_filter(c, wall_id, cx, cy, left, right, top, bottom):
            continue
        openings.append(contour_to_opening(c, "door", wall_id, left, right, top, bottom, mm_per_px, room_len_mm, room_wid_mm))

    for c in win_cnts:
        m = cv2.moments(c)
        if m["m00"] == 0:
            continue
        cx = m["m10"] / m["m00"]
        cy = m["m01"] / m["m00"]
        wall_id = assign_wall(cx, cy, left, right, top, bottom)
        if not contour_passes_opening_shape_filter(c, wall_id, cx, cy, left, right, top, bottom):
            continue
        openings.append(contour_to_opening(c, "window", wall_id, left, right, top, bottom, mm_per_px, room_len_mm, room_wid_mm))

    if openings:
        max_w = max(o["width"] for o in openings)
        if len(openings) > 8 or max_w > int(0.35 * max(room_len_mm, room_wid_mm)):
            openings = []

    openings.sort(key=lambda o: (o["wall_id"], o["offset_from_start"], o["type"]))
    door_i = 0
    win_i = 0
    for o in openings:
        if o["type"] == "door":
            door_i += 1
            o["id"] = f"D{door_i}"
        else:
            win_i += 1
            o["id"] = f"WIN{win_i}"

    state = {
        "units": "mm",
        "space": {
            "name": "Room",
            "height": DEFAULT_CEILING_MM,
            "footprint": {"min_x": 0, "min_y": 0, "max_x": room_len_mm, "max_y": room_wid_mm},
        },
        "exterior_walls": build_wall_defs(room_len_mm, room_wid_mm),
        "openings": openings,
        "columns": [],
        "inner_walls": inner_walls,
        "beams": [],
        "services": {
            "points": [
                {
                    "id": obj["id"],
                    "service_type": obj["type"],
                    "footprint": obj["footprint"],
                }
                for obj in objects
            ]
        },
        "objects": objects,
        "rooms": rooms,
        "text_annotations": [{"text": t["text"], "x": t["x"], "y": t["y"], "w": t["w"], "h": t["h"], "conf": round(float(t["conf"]), 2)} for t in tokens],
        "meta": {
            "door_counter": door_i,
            "window_counter": win_i,
            "scale_mm_per_px": round(float(mm_per_px), 4),
            "ocr_tokens": len(tokens),
            "detected_room_labels": len(rooms),
            "detected_objects": len(objects),
            "ocr_engine": "google-vision",
        },
    }
    return state

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fast floor plan extraction using Google Cloud Vision")
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--json-out", default="floorplan_state.json", help="Output JSON file")
    parser.add_argument("--svg-out", default="final.svg", help="Output SVG file")
    parser.add_argument("--credentials", default=None, help="Path to Google Cloud credentials JSON")
    args = parser.parse_args()

    if args.credentials:
        import os
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.credentials

    state = extract_floorplan_fast(Path(args.image))
    Path(args.json_out).write_text(json.dumps(state, indent=2))
    write_svg(state, Path(args.svg_out))

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.svg_out}")
    print(f"Rooms: {len(state.get('rooms', []))}, Scale: {state.get('meta', {}).get('scale_mm_per_px')} mm/px")
