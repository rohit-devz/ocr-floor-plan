import re


ROOM_KEYWORDS = [
    "KITCHEN",
    "BEDROOM",
    "LIVING",
    "DINING",
    "BATH",
    "BALCONY",
    "TOILET",
    "LAUNDRY",
    "CLOSET",
    "UNIT",
]

DIMENSION_PATTERN = re.compile(
    r"(\d{1,2})\s*['\"]?\s*(\d{0,2})?\s*['\"]?\s*[xX]\s*(\d{1,2})\s*['\"]?\s*(\d{0,2})?",
    re.IGNORECASE,
)
AREA_PATTERN = re.compile(
    r"(\d{1,4}(?:\.\d+)?)\s*(?:SQ\.?\s*FT|SQFT|SFT|FT2|FT\^2)",
    re.IGNORECASE,
)


def normalize_dimension(match: re.Match) -> dict:
    ft1, in1, ft2, in2 = match.groups()
    return {
        "width": {"feet": int(ft1), "inches": int(in1) if in1 and in1.isdigit() else 0},
        "height": {
            "feet": int(ft2),
            "inches": int(in2) if in2 and in2.isdigit() else 0,
        },
    }


def extract_area_candidates(line: str) -> list[float]:
    return [float(m.group(1)) for m in AREA_PATTERN.finditer(line)]


def find_room(line: str) -> str | None:
    for keyword in ROOM_KEYWORDS:
        if keyword in line:
            return keyword
    return None


def pick_closest_area_to_room(line: str, room_name: str, areas: list[float]) -> float | None:
    if not areas:
        return None
    room_pos = line.find(room_name)
    if room_pos < 0:
        return areas[0]

    matches = list(AREA_PATTERN.finditer(line))
    if not matches:
        return areas[0]

    best_idx = 0
    best_dist = 10**9
    for idx, m in enumerate(matches):
        center = (m.start() + m.end()) // 2
        dist = abs(center - room_pos)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return areas[best_idx]


def organize_floor_data(text1: str, text2: str | None = None) -> dict:
    texts = [text1]
    if text2:
        texts.append(text2)

    combined_text = "\n".join(texts).upper()
    lines = [line.strip() for line in combined_text.splitlines() if line.strip()]

    rooms: list[dict] = []
    current_room: dict | None = None

    for idx, line in enumerate(lines):
        dim_match = DIMENSION_PATTERN.search(line)
        area_candidates = extract_area_candidates(line)
        room_found = find_room(line)

        if room_found:
            current_room = {
                "name": room_found,
                "raw_label": line,
                "dimensions": None,
                "area": None,
                "area_candidates_sqft": area_candidates.copy(),
            }
            rooms.append(current_room)

            if dim_match:
                current_room["dimensions"] = normalize_dimension(dim_match)

            lookahead_candidates = area_candidates.copy()
            for j in range(idx + 1, min(len(lines), idx + 4)):
                nxt = lines[j]
                if find_room(nxt):
                    break
                lookahead_candidates.extend(extract_area_candidates(nxt))

                if current_room["dimensions"] is None:
                    nxt_dim = DIMENSION_PATTERN.search(nxt)
                    if nxt_dim:
                        current_room["dimensions"] = normalize_dimension(nxt_dim)

            current_room["area_candidates_sqft"] = lookahead_candidates
            if lookahead_candidates:
                if room_found == "KITCHEN":
                    picked = max(lookahead_candidates)
                else:
                    picked = lookahead_candidates[0]
                current_room["area"] = {"sqft": picked}
            continue

        if current_room is None:
            continue

        if dim_match and current_room["dimensions"] is None:
            current_room["dimensions"] = normalize_dimension(dim_match)

    return {
        "unit": next((line for line in lines if line.startswith("UNIT")), None),
        "rooms": rooms,
    }
