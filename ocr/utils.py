import json
import re
from collections import defaultdict

ROOM_KEYWORDS = [
    "BALCONY",
    "KITCHEN",
    "BEDROOM",
    "LIVING",
    "BATH",
    "LAUNDRY",
    "WALK-IN",
    "CLOSET",
    "UNIT",
]

DIMENSION_PATTERN = re.compile(
    r"(\d{1,2})\s*['\"]?\s*(\d{0,2})?\s*['\"]?\s*[xX]\s*(\d{1,2})\s*['\"]?\s*(\d{0,2})?",
    re.IGNORECASE,
)


def normalize_dimension(match):
    ft1, in1, ft2, in2 = match.groups()
    return {
        "width": {"feet": int(ft1), "inches": int(in1) if in1 and in1.isdigit() else 0},
        "height": {
            "feet": int(ft2),
            "inches": int(in2) if in2 and in2.isdigit() else 0,
        },
    }


def organize_floor_data(text1: str, text2: str | None = None) -> dict:
    texts = [text1]

    if text2:
        texts.append(text2)

    combined_text = "\n".join(texts).upper()
    lines = [line.strip() for line in combined_text.splitlines() if line.strip()]

    # rest of your logic...
    rooms = []
    current_room = None

    for line in lines:
        dim_match = DIMENSION_PATTERN.search(line)

        # Check if this line contains a room keyword
        room_found = None
        for keyword in ROOM_KEYWORDS:
            if keyword in line:
                room_found = keyword
                break

        # If we found a room name
        if room_found:
            current_room = {"name": room_found, "raw_label": line, "dimensions": None}
            rooms.append(current_room)

            # If dimensions are on the SAME line, attach immediately
            if dim_match:
                current_room["dimensions"] = normalize_dimension(dim_match)
            continue

        # If line is only dimensions, attach to last room
        if dim_match and current_room and current_room["dimensions"] is None:
            current_room["dimensions"] = normalize_dimension(dim_match)

    return {
        "unit": next((line for line in lines if line.startswith("UNIT")), None),
        "rooms": rooms,
    }
