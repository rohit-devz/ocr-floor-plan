from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

import cv2
import numpy as np

DEFAULT_CEILING_MM = 3048
DEFAULT_WALL_THICKNESS_MM = 150
DEFAULT_DOOR_HEIGHT_MM = 2133
DEFAULT_WINDOW_HEIGHT_MM = 1219
DEFAULT_WINDOW_SILL_MM = 914
DEFAULT_MM_PER_PX = 5.0

ROOM_KEYWORDS = [
    "BEDROOM",
    "KITCHEN",
    "TOILET",
    "DINING",
    "DRAWING",
    "LIVING",
    "DRESS",
    "BALCONY",
    "UTILITY",
    "FOYER",
    "SERVANT",
    "DRG",
]

FIXTURE_KEYWORDS = {
    "SINK": "sink",
    "BASIN": "wash_basin",
    "WASH": "wash_area",
    "STOVE": "stove",
    "GAS": "gas_stove",
    "HOB": "hob",
    "COOK": "cooktop",
    "WC": "wc",
    "TOILET": "toilet_fixture",
    "SHOWER": "shower",
    "FRIDGE": "fridge",
    "REFRIGERATOR": "fridge",
    "CHIMNEY": "chimney",
}


def clamp_int(v: float, lo: int, hi: int) -> int:
    return int(max(lo, min(hi, round(v))))


def px_to_mm(v: float, mm_per_px: float) -> int:
    return int(round(v * mm_per_px))


def normalize_line(line: np.ndarray, axis_tol: int = 4) -> tuple[str, float, float, float] | None:
    x1, y1, x2, y2 = [float(v) for v in line]
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    if dy <= axis_tol and dx > axis_tol:
        y = (y1 + y2) / 2.0
        a0, a1 = sorted([x1, x2])
        return ("h", y, a0, a1)
    if dx <= axis_tol and dy > axis_tol:
        x = (x1 + x2) / 2.0
        a0, a1 = sorted([y1, y2])
        return ("v", x, a0, a1)
    return None


def merge_axis_lines(lines: list[tuple[str, float, float, float]], coord_tol: int = 8, gap_tol: int = 16) -> list[dict[str, float]]:
    clusters: list[dict[str, Any]] = []
    for ori, c, a0, a1 in lines:
        merged = False
        for cl in clusters:
            if cl["ori"] != ori:
                continue
            if abs(cl["coord"] - c) > coord_tol:
                continue
            if a1 < cl["a0"] - gap_tol or a0 > cl["a1"] + gap_tol:
                continue
            cl["a0"] = min(cl["a0"], a0)
            cl["a1"] = max(cl["a1"], a1)
            cl["coord_sum"] += c
            cl["count"] += 1
            cl["coord"] = cl["coord_sum"] / cl["count"]
            merged = True
            break
        if not merged:
            clusters.append({"ori": ori, "coord": c, "a0": a0, "a1": a1, "coord_sum": c, "count": 1})

    out: list[dict[str, float]] = []
    for cl in clusters:
        out.append(
            {
                "ori": cl["ori"],
                "coord": float(cl["coord"]),
                "a0": float(cl["a0"]),
                "a1": float(cl["a1"]),
                "length": float(cl["a1"] - cl["a0"]),
            }
        )
    return out


def detect_lines(gray: np.ndarray) -> list[dict[str, float]]:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    edges = cv2.Canny(thr, 50, 150)
    min_line = max(50, int(min(gray.shape[:2]) * 0.12))
    raw = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=80, minLineLength=min_line, maxLineGap=12)
    if raw is None:
        return []
    norm: list[tuple[str, float, float, float]] = []
    for r in raw[:, 0, :]:
        n = normalize_line(r)
        if n is not None:
            norm.append(n)
    return merge_axis_lines(norm)


def detect_outer_frame(merged_lines: list[dict[str, float]], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    hs = [l for l in merged_lines if l["ori"] == "h"]
    vs = [l for l in merged_lines if l["ori"] == "v"]
    if not hs or not vs:
        return 0.0, float(img_w - 1), 0.0, float(img_h - 1)

    hmax = max(l["length"] for l in hs)
    vmax = max(l["length"] for l in vs)
    h_candidates = [l for l in hs if l["length"] >= 0.45 * hmax] or hs
    v_candidates = [l for l in vs if l["length"] >= 0.45 * vmax] or vs

    top = min(h_candidates, key=lambda l: l["coord"])["coord"]
    bottom = max(h_candidates, key=lambda l: l["coord"])["coord"]
    left = min(v_candidates, key=lambda l: l["coord"])["coord"]
    right = max(v_candidates, key=lambda l: l["coord"])["coord"]
    return left, right, top, bottom


def detect_room_components(gray: np.ndarray) -> list[dict[str, float]]:
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    inv = cv2.bitwise_not(thr)
    n, _labels, stats, _cent = cv2.connectedComponentsWithStats(inv, connectivity=8)
    h, w = gray.shape[:2]
    area_img = h * w

    comps: list[dict[str, float]] = []
    for i in range(1, n):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        rw = int(stats[i, cv2.CC_STAT_WIDTH])
        rh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        if area < area_img * 0.002 or area > area_img * 0.5:
            continue
        if x <= 1 or y <= 1 or x + rw >= w - 1 or y + rh >= h - 1:
            continue

        comps.append({"x": x, "y": y, "w": rw, "h": rh, "cx": x + rw / 2.0, "cy": y + rh / 2.0, "area": area})

    comps.sort(key=lambda c: c["area"], reverse=True)
    return comps


def _dedupe_tokens(tokens: list[dict[str, Any]], grid: int = 6) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen = set()
    for t in tokens:
        key = (
            str(t.get("text", "")).upper(),
            round(float(t.get("x", 0)) / grid),
            round(float(t.get("y", 0)) / grid),
            round(float(t.get("w", 0)) / grid),
            round(float(t.get("h", 0)) / grid),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(t)
    return out


def _ocr_tokens_tesseract(image: np.ndarray, tessdata_dir: str | None = None) -> list[dict[str, Any]]:
    try:
        import pytesseract
    except Exception:
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2x = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    blur = cv2.GaussianBlur(gray2x, (3, 3), 0)
    _, otsu = cv2.threshold(gray2x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adap = cv2.adaptiveThreshold(
        gray2x,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        7,
    )
    inv = cv2.bitwise_not(gray2x)

    cfg_base = "--oem 1"
    if tessdata_dir:
        cfg_base += f" --tessdata-dir {tessdata_dir}"

    passes = [
        (image, 1.0, cfg_base + " --psm 11"),
        (image, 1.0, cfg_base + " --psm 6"),
        (gray2x, 2.0, cfg_base + " --psm 11"),
        (gray2x, 2.0, cfg_base + " --psm 6"),
        (blur, 2.0, cfg_base + " --psm 6"),
        (otsu, 2.0, cfg_base + " --psm 6"),
        (adap, 2.0, cfg_base + " --psm 6"),
        (inv, 2.0, cfg_base + " --psm 11"),
    ]

    out: list[dict[str, Any]] = []
    for oimg, scale, cfg in passes:
        data = None
        for lang in ("eng", "osd", "eng+osd"):
            try:
                data = pytesseract.image_to_data(oimg, output_type=pytesseract.Output.DICT, config=cfg, lang=lang)
                break
            except Exception:
                data = None
        if data is None:
            continue

        for text, conf, x, y, w, h in zip(
            data.get("text", []),
            data.get("conf", []),
            data.get("left", []),
            data.get("top", []),
            data.get("width", []),
            data.get("height", []),
        ):
            t = (text or "").strip()
            if not t:
                continue
            try:
                c = float(conf)
            except Exception:
                c = -1
            if c < 5:
                continue
            rx = int(round(int(x) / scale))
            ry = int(round(int(y) / scale))
            rw = int(round(int(w) / scale))
            rh = int(round(int(h) / scale))
            out.append({"text": t, "conf": c, "x": rx, "y": ry, "w": rw, "h": rh, "cx": rx + rw / 2.0, "cy": ry + rh / 2.0})

    return _dedupe_tokens(out)


def _ocr_tokens_easyocr(image: np.ndarray) -> list[dict[str, Any]]:
    try:
        import easyocr
    except Exception:
        return []

    reader = easyocr.Reader(["en"], gpu=False)
    # Preprocess: convert to grayscale and upscale 2x to improve small-text detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray2x = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    _easyocr_scale = 2.0
    results = reader.readtext(gray2x, detail=1)
    out: list[dict[str, Any]] = []
    for item in results:
        if len(item) != 3:
            continue
        box, text, conf = item
        t = str(text or "").strip()
        if not t:
            continue
        try:
            c = float(conf) * 100.0
        except Exception:
            c = -1.0
        if c < 5:
            continue
        # Scale bounding box coordinates back to original image space
        xs = [float(pt[0]) / _easyocr_scale for pt in box]
        ys = [float(pt[1]) / _easyocr_scale for pt in box]
        x0, x1 = int(round(min(xs))), int(round(max(xs)))
        y0, y1 = int(round(min(ys))), int(round(max(ys)))
        w = max(1, x1 - x0)
        h = max(1, y1 - y0)
        out.append({"text": t, "conf": c, "x": x0, "y": y0, "w": w, "h": h, "cx": x0 + w / 2.0, "cy": y0 + h / 2.0})

    return _dedupe_tokens(out)


def _ocr_tokens_paddle(image: np.ndarray) -> list[dict[str, Any]]:
    try:
        from paddleocr import PaddleOCR
    except Exception:
        return []

    try:
        ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        results = ocr.ocr(image, cls=True)
    except Exception:
        return []

    out: list[dict[str, Any]] = []
    for page in results or []:
        for det in page or []:
            if not isinstance(det, (list, tuple)) or len(det) != 2:
                continue
            box, rec = det
            if not rec or len(rec) < 2:
                continue
            text = str(rec[0] or "").strip()
            if not text:
                continue
            try:
                c = float(rec[1]) * 100.0
            except Exception:
                c = -1.0
            if c < 5:
                continue
            try:
                xs = [float(pt[0]) for pt in box]
                ys = [float(pt[1]) for pt in box]
            except Exception:
                continue
            x0, x1 = int(round(min(xs))), int(round(max(xs)))
            y0, y1 = int(round(min(ys))), int(round(max(ys)))
            w = max(1, x1 - x0)
            h = max(1, y1 - y0)
            out.append({"text": text, "conf": c, "x": x0, "y": y0, "w": w, "h": h, "cx": x0 + w / 2.0, "cy": y0 + h / 2.0})

    return _dedupe_tokens(out)


def ocr_tokens(image: np.ndarray, tessdata_dir: str | None = None, engine: str = "paddle") -> list[dict[str, Any]]:
    e = (engine or "paddle").strip().lower()
    if e == "paddle":
        t = _ocr_tokens_paddle(image)
        if t:
            return t
        t = _ocr_tokens_easyocr(image)
        if t:
            return t
        return _ocr_tokens_tesseract(image, tessdata_dir=tessdata_dir)
    if e == "easyocr":
        t = _ocr_tokens_easyocr(image)
        if t:
            return t
        return _ocr_tokens_tesseract(image, tessdata_dir=tessdata_dir)
    return _ocr_tokens_tesseract(image, tessdata_dir=tessdata_dir)


def is_room_label_token(text: str) -> bool:
    up = re.sub(r"[^A-Z]", "", text.upper())
    if len(up) < 4:
        return False
    return any(k in up for k in ROOM_KEYWORDS)


def normalize_label_name(text: str) -> str:
    up = re.sub(r"[^A-Z ]", "", text.upper()).strip()
    if "BEDROOM" in up:
        return "BEDROOM"
    if "KITCHEN" in up or "ITCHEN" in up:
        return "KITCHEN"
    if "TOILET" in up:
        return "TOILET"
    if "DINING" in up or "DRG" in up or "DRAWING" in up:
        return "DRG / DINING"
    if "LIVING" in up:
        return "LIVING"
    if "BALCONY" in up:
        return "BALCONY"
    return up if up else "ROOM"


def parse_feet_inches_single(token: str) -> float | None:
    t = token.upper().replace("\u201d", '"').replace("\u2032", "'").replace(" ", "")
    t = t.replace("`", "'").replace("’", "'").replace("″", '"').replace("“", '"')
    # OCR cleanup: 13'-6" -> 13'6", 9'-O" -> 9'0"
    t = t.replace("'-", "'").replace("-'", "'")
    t = t.replace("O", "0").replace("Q", "0").replace("D", "0")
    t = t.replace("I", "1").replace("L", "1")
    t = re.sub(r"[^0-9'\"]", "", t)

    # 13'6" / 13'06" / 13'6
    m = re.search(r"^(\d{1,2})'(\d{1,2})\"?$", t)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2))
        if 0 <= inch < 12:
            return ft + (inch / 12.0)

    # 13-6 / 13'6 (already cleaned)
    m = re.search(r"^(\d{1,2})[-'](\d{1,2})\"?$", t)
    if m:
        ft = int(m.group(1))
        inch = int(m.group(2))
        if 0 <= inch < 12:
            return ft + (inch / 12.0)

    m = re.search(r"^(\d{2})$", t)
    if m:
        d = m.group(1)
        ft = int(d[0])
        inch = int(d[1])
        if 0 <= inch < 12:
            return ft + (inch / 12.0)

    m = re.search(r"^(\d{1,2})\"?$", t)
    if m:
        ft = int(m.group(1))
        if 3 <= ft <= 30:
            return float(ft)

    m = re.search(r"^(\d{1,2})'$", t)
    if m:
        ft = int(m.group(1))
        if 3 <= ft <= 30:
            return float(ft)

    return None


def parse_dimension_from_text(text: str) -> tuple[float, float] | None:
    s = text.upper().replace(" ", "")
    s = s.replace("\u00D7", "X").replace("*", "X").replace("/", "X")
    s = s.replace("O", "0").replace("Q", "0").replace("D", "0")
    s = s.replace("”", '"').replace("′", "'").replace("`", "'")
    s = s.replace("'-", "'").replace("-'", "'")
    # Common OCR misreads near inch markers.
    s = re.sub(r"([0-9]')([A-Z]{1,2})(\")", r"\g<1>0\g<3>", s)
    s = re.sub(r"([0-9]')([A-Z]{1,2})$", r"\g<1>0", s)

    token = r"(?:\d{1,2}(?:['-]\d{1,2})?\"?|\d{1,2}'?)"
    patterns = [
        re.compile(rf"({token})X({token})"),
        re.compile(r'([0-9\-\'\"]+)X([0-9\-\'\"]+)'),
    ]

    candidates: list[tuple[float, float]] = []
    for pat in patterns:
        for m in pat.finditer(s):
            a = parse_feet_inches_single(m.group(1))
            b = parse_feet_inches_single(m.group(2))
            if a is None or b is None:
                continue
            if not (2.5 <= a <= 40 and 2.5 <= b <= 40):
                continue
            candidates.append((a, b))

    if not candidates and "X" in s:
        parts = [p for p in re.split(r"X+", s) if p]
        for i in range(len(parts) - 1):
            a = parse_feet_inches_single(parts[i])
            b = parse_feet_inches_single(parts[i + 1])
            if a is None or b is None:
                continue
            if not (2.5 <= a <= 40 and 2.5 <= b <= 40):
                continue
            candidates.append((a, b))

    if not candidates:
        return None

    # Prefer realistic room dimensions and avoid huge OCR outliers.
    def score(pair: tuple[float, float]) -> tuple[int, float]:
        a, b = pair
        mx = max(a, b)
        realism = 0
        if mx <= 25:
            realism += 2
        if min(a, b) >= 4:
            realism += 1
        return (realism, -(abs(a - b)))

    return sorted(candidates, key=score, reverse=True)[0]


def _bbox_iou_like(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0 = max(ax0, bx0)
    iy0 = max(ay0, by0)
    ix1 = min(ax1, bx1)
    iy1 = min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = float((ix1 - ix0) * (iy1 - iy0))
    aarea = float(max(1, (ax1 - ax0) * (ay1 - ay0)))
    return inter / aarea


def clamp_rooms_to_space(rooms: list[dict[str, Any]], max_x: int, max_y: int) -> None:
    for r in rooms:
        fp = r.get("footprint", {})
        min_x = clamp_int(fp.get("min_x", 0), 0, max_x)
        max_xv = clamp_int(fp.get("max_x", max_x), 0, max_x)
        min_y = clamp_int(fp.get("min_y", 0), 0, max_y)
        max_yv = clamp_int(fp.get("max_y", max_y), 0, max_y)
        if max_xv <= min_x:
            max_xv = min(max_x, min_x + 1)
        if max_yv <= min_y:
            max_yv = min(max_y, min_y + 1)
        fp["min_x"] = min_x
        fp["max_x"] = max_xv
        fp["min_y"] = min_y
        fp["max_y"] = max_yv
        r["footprint"] = fp




def filter_invalid_rooms(rooms: list[dict[str, Any]], min_side_mm: int = 120) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for r in rooms:
        fp = r.get("footprint", {})
        w = int(fp.get("max_x", 0) - fp.get("min_x", 0))
        h = int(fp.get("max_y", 0) - fp.get("min_y", 0))
        # If room has explicit dimensions from OCR, trust them over computed footprint
        has_ocr_dims = "dimensions_ft" in r
        if (w < min_side_mm or h < min_side_mm) and not has_ocr_dims:
            continue
        out.append(r)
    for i, r in enumerate(out, 1):
        r["id"] = i
    return out

def merge_room_sets(primary: list[dict[str, Any]], secondary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    Merge room candidates from multiple detectors.
    Keeps primary first, then adds missing semantic labels from secondary.
    """
    out = list(primary)
    existing_keys = set()
    for r in out:
        key = normalize_label_name(str(r.get("label_text", r.get("name", ""))))
        existing_keys.add(key)

    for r in secondary:
        key = normalize_label_name(str(r.get("label_text", r.get("name", ""))))
        # Add if label class missing entirely, or if this room has dimensions while
        # existing class entries do not.
        if key not in existing_keys:
            out.append(r)
            existing_keys.add(key)
            continue
        if r.get("dimensions_ft"):
            has_dim_for_key = False
            for ex in out:
                ex_key = normalize_label_name(str(ex.get("label_text", ex.get("name", ""))))
                if ex_key == key and ex.get("dimensions_ft"):
                    has_dim_for_key = True
                    break
            if not has_dim_for_key:
                out.append(r)

    # Re-number IDs after merge.
    for i, r in enumerate(out, 1):
        r["id"] = i
    return out


def group_tokens_into_lines(tokens: list[dict[str, Any]], y_tol: int = 10) -> list[dict[str, Any]]:
    if not tokens:
        return []

    ts = sorted(tokens, key=lambda t: (t["cy"], t["x"]))
    groups: list[list[dict[str, Any]]] = []

    for t in ts:
        placed = False
        for g in groups:
            gy = sum(x["cy"] for x in g) / len(g)
            if abs(t["cy"] - gy) <= y_tol:
                g.append(t)
                placed = True
                break
        if not placed:
            groups.append([t])

    lines: list[dict[str, Any]] = []
    for g in groups:
        g.sort(key=lambda x: x["x"])
        text = " ".join(x["text"] for x in g)
        x0 = min(x["x"] for x in g)
        y0 = min(x["y"] for x in g)
        x1 = max(x["x"] + x["w"] for x in g)
        y1 = max(x["y"] + x["h"] for x in g)
        lines.append({"text": text, "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0, "cx": (x0 + x1) / 2.0, "cy": (y0 + y1) / 2.0})

    lines.sort(key=lambda l: (l["y"], l["x"]))
    return lines


def _extract_dimension_near_label(tokens: list[dict[str, Any]], label: dict[str, Any]) -> tuple[float, float] | None:
    # Adaptive search window: larger plans often place dimensions a bit farther
    # under room labels.
    x_pad_l = max(60, int(label.get("w", 0) * 1.1))
    x_pad_r = max(260, int(label.get("w", 0) * 2.8))
    y_pad_up = 40
    y_pad_down = 220

    near = []
    for t in tokens:
        if t["cy"] < label["cy"] - y_pad_up or t["cy"] > label["cy"] + y_pad_down:
            continue
        if t["x"] < label["x"] - x_pad_l or t["x"] > label["x"] + x_pad_r:
            continue
        near.append(t)

    if not near:
        return None

    near.sort(key=lambda z: (z["y"], z["x"]))
    near_lines = group_tokens_into_lines(near, y_tol=12)

    # Candidate tuples: (dim_pair, cx, cy)
    candidates: list[tuple[tuple[float, float], float, float]] = []

    # 1) Text-window candidates.
    texts = [t["text"] for t in near]
    text_cands: list[tuple[str, float, float]] = []
    for t in near:
        text_cands.append((t["text"], t["cx"], t["cy"]))
    for i in range(len(texts)):
        for j in range(i + 1, min(len(texts), i + 5)):
            merged = near[i : j + 1]
            cx = float(sum(z["cx"] for z in merged) / len(merged))
            cy = float(sum(z["cy"] for z in merged) / len(merged))
            text_cands.append(("".join(texts[i : j + 1]), cx, cy))
            text_cands.append((" ".join(texts[i : j + 1]), cx, cy))
    for ln in near_lines:
        text_cands.append((ln["text"], ln["cx"], ln["cy"]))

    for c, cx, cy in text_cands:
        d = parse_dimension_from_text(c)
        if d is not None:
            candidates.append((d, cx, cy))

    # 2) Geometry candidates: nearest numeric tokens around an X separator.
    by_line = group_tokens_into_lines(near, y_tol=9)
    for ln in by_line:
        ltoks = [t for t in near if abs(t["cy"] - ln["cy"]) <= 12 and t["x"] >= ln["x"] - 4 and t["x"] <= (ln["x"] + ln["w"] + 4)]
        ltoks.sort(key=lambda z: z["x"])
        if not ltoks:
            continue
        for x_tok in ltoks:
            xu = re.sub(r"[^A-Z]", "", x_tok["text"].upper())
            if "X" not in xu:
                continue
            left_num = None
            right_num = None
            best_left_dx = 1e9
            best_right_dx = 1e9
            for t in ltoks:
                val = parse_feet_inches_single(t["text"])
                if val is None:
                    continue
                dx = abs(t["cx"] - x_tok["cx"])
                if t["cx"] < x_tok["cx"] and dx < best_left_dx:
                    best_left_dx = dx
                    left_num = (val, t)
                if t["cx"] > x_tok["cx"] and dx < best_right_dx:
                    best_right_dx = dx
                    right_num = (val, t)
            if left_num and right_num and best_left_dx <= 130 and best_right_dx <= 130:
                d = (float(left_num[0]), float(right_num[0]))
                cx = float((left_num[1]["cx"] + right_num[1]["cx"]) / 2.0)
                cy = float((left_num[1]["cy"] + right_num[1]["cy"]) / 2.0)
                candidates.append((d, cx, cy))

    if not candidates:
        return None

    best = None
    best_score = 1e9
    for d, cx, cy in candidates:
        dist = float(np.hypot((cx - label["cx"]) * 0.65, max(0.0, cy - label["cy"])))
        label_name = normalize_label_name(label["text"])
        penalty = 0.0
        if label_name == "TOILET" and max(d[0], d[1]) > 9.5:
            continue
        if label_name != "TOILET" and min(d[0], d[1]) < 4.0:
            continue
        if max(d[0], d[1]) > 25:
            penalty += 20.0
        if min(d[0], d[1]) < 4.8:
            penalty += 8.0
        score = dist + penalty
        if score < best_score:
            best_score = score
            best = d

    return best


def build_labeled_rooms(tokens: list[dict[str, Any]], components: list[dict[str, float]], mm_per_px: float) -> list[dict[str, Any]]:
    labels = [t for t in tokens if is_room_label_token(t["text"])]
    # De-dup similar nearby labels (multi-pass OCR duplicates).
    dedup: list[dict[str, Any]] = []
    for t in labels:
        same = False
        for e in dedup:
            if normalize_label_name(e["text"]) != normalize_label_name(t["text"]):
                continue
            if abs(e["x"] - t["x"]) <= 20 and abs(e["y"] - t["y"]) <= 20:
                same = True
                break
        if not same:
            dedup.append(t)
    labels = dedup

    rooms: list[dict[str, Any]] = []
    used_comp = set()
    name_counter: dict[str, int] = {}

    # First pass: component-based matching
    for lb in labels:
        best_dim = _extract_dimension_near_label(tokens, lb)

        best_comp_idx = None
        best_cd = 1e9
        for idx, c in enumerate(components):
            if idx in used_comp:
                continue
            cd = float(np.hypot(c["cx"] - lb["cx"], c["cy"] - lb["cy"]))
            if cd < best_cd:
                best_cd = cd
                best_comp_idx = idx

        if best_comp_idx is None:
            continue
        used_comp.add(best_comp_idx)
        c = components[best_comp_idx]

        raw_name = lb["text"].strip()
        base = normalize_label_name(raw_name)
        k = base.upper()
        name_counter[k] = name_counter.get(k, 0) + 1
        name = f"{base} {name_counter[k]}" if name_counter[k] > 1 else base

        room: dict[str, Any] = {
            "id": len(rooms) + 1,
            "name": name,
            "label_text": raw_name,
            "footprint": {
                "min_x": px_to_mm(c["x"], mm_per_px),
                "min_y": px_to_mm(c["y"], mm_per_px),
                "max_x": px_to_mm(c["x"] + c["w"], mm_per_px),
                "max_y": px_to_mm(c["y"] + c["h"], mm_per_px),
            },
        }

        if best_dim is not None:
            room["dimensions_ft"] = {"length": round(best_dim[0], 2), "width": round(best_dim[1], 2)}

        rooms.append(room)


    return rooms


def build_labeled_rooms_from_lines(
    tokens: list[dict[str, Any]],
    merged_lines: list[dict[str, float]],
    left: float,
    right: float,
    top: float,
    bottom: float,
    mm_per_px: float,
) -> list[dict[str, Any]]:
    labels = [t for t in tokens if is_room_label_token(t["text"])]
    dedup: list[dict[str, Any]] = []
    for t in labels:
        same = False
        for e in dedup:
            if normalize_label_name(e["text"]) != normalize_label_name(t["text"]):
                continue
            if abs(e["x"] - t["x"]) <= 20 and abs(e["y"] - t["y"]) <= 20:
                same = True
                break
        if not same:
            dedup.append(t)
    labels = dedup

    rooms: list[dict[str, Any]] = []
    name_counter: dict[str, int] = {}

    for lb in labels:
        x = lb["cx"]
        y = lb["cy"]

        v_hits = [l for l in merged_lines if l["ori"] == "v" and (l["a0"] - 8) <= y <= (l["a1"] + 8)]
        h_hits = [l for l in merged_lines if l["ori"] == "h" and (l["a0"] - 8) <= x <= (l["a1"] + 8)]

        l_candidates = [left] + [v["coord"] for v in v_hits if v["coord"] < x - 2]
        r_candidates = [right] + [v["coord"] for v in v_hits if v["coord"] > x + 2]
        t_candidates = [top] + [h["coord"] for h in h_hits if h["coord"] < y - 2]
        b_candidates = [bottom] + [h["coord"] for h in h_hits if h["coord"] > y + 2]

        lx = max(l_candidates) if l_candidates else left
        rx = min(r_candidates) if r_candidates else right
        ty = max(t_candidates) if t_candidates else top
        by = min(b_candidates) if b_candidates else bottom

        if rx - lx < 25 or by - ty < 25:
            # fallback small box around label
            lx = max(left, x - 80)
            rx = min(right, x + 80)
            ty = max(top, y - 60)
            by = min(bottom, y + 60)

        base = normalize_label_name(lb["text"])
        k = base.upper()
        name_counter[k] = name_counter.get(k, 0) + 1
        name = f"{base} {name_counter[k]}" if name_counter[k] > 1 else base

        room = {
            "id": len(rooms) + 1,
            "name": name,
            "label_text": lb["text"],
            "footprint": {
                "min_x": px_to_mm(lx - left, mm_per_px),
                "max_x": px_to_mm(rx - left, mm_per_px),
                "min_y": px_to_mm(bottom - by, mm_per_px),
                "max_y": px_to_mm(bottom - ty, mm_per_px),
            },
        }

        d = _extract_dimension_near_label(tokens, lb)
        if d is not None:
            room["dimensions_ft"] = {"length": round(d[0], 2), "width": round(d[1], 2)}

        rooms.append(room)

    # de-dup identical footprints
    cleaned: list[dict[str, Any]] = []
    for r in rooms:
        fp = r["footprint"]
        dup = False
        for e in cleaned:
            ef = e["footprint"]
            if (
                abs(fp["min_x"] - ef["min_x"]) <= 40
                and abs(fp["max_x"] - ef["max_x"]) <= 40
                and abs(fp["min_y"] - ef["min_y"]) <= 40
                and abs(fp["max_y"] - ef["max_y"]) <= 40
                and normalize_label_name(r["label_text"]) == normalize_label_name(e["label_text"])
            ):
                dup = True
                break
        if not dup:
            cleaned.append(r)
    return cleaned


def calibrate_mm_per_px_from_rooms(rooms: list[dict[str, Any]], components: list[dict[str, float]], fallback: float) -> float:
    if not rooms:
        return fallback

    cands = []
    for r in rooms:
        d = r.get("dimensions_ft")
        fp = r.get("footprint", {})
        if not d or not fp:
            continue

        w_mm = float((fp.get("max_x", 0) - fp.get("min_x", 0)))
        h_mm = float((fp.get("max_y", 0) - fp.get("min_y", 0)))
        if w_mm <= 0 or h_mm <= 0:
            continue

        # Reverse prior scale from mm to pseudo-px using fallback; then estimate corrected mm/px.
        w_px = w_mm / fallback
        h_px = h_mm / fallback
        if w_px <= 1 or h_px <= 1:
            continue

        dim_a = float(d["length"]) * 304.8
        dim_b = float(d["width"]) * 304.8

        px_small, px_large = sorted([w_px, h_px])
        mm_small, mm_large = sorted([dim_a, dim_b])

        cands.append(mm_small / px_small)
        cands.append(mm_large / px_large)

    cands = [c for c in cands if 0.8 <= c <= 20.0]
    if len(cands) < 4:
        return fallback
    arr = np.array(cands, dtype=float)
    med = float(np.median(arr))
    if med <= 0:
        return fallback
    mad = float(np.median(np.abs(arr - med)))
    rel_mad = mad / med if med > 0 else 1.0
    # Accept only consistent calibrations; otherwise preserve fallback.
    if rel_mad > 0.28:
        return fallback
    return med


def build_inner_walls(merged_lines: list[dict[str, float]], left: float, right: float, top: float, bottom: float, mm_per_px: float) -> list[dict[str, Any]]:
    room_w_px = right - left
    room_h_px = bottom - top
    min_len = max(40.0, 0.18 * min(room_w_px, room_h_px))

    thick_px = max(3.0, DEFAULT_WALL_THICKNESS_MM / mm_per_px)
    half_t = thick_px / 2.0

    inner: list[dict[str, Any]] = []
    idx = 1

    for l in merged_lines:
        if l["length"] < min_len:
            continue

        ori = l["ori"]
        c = l["coord"]
        a0 = l["a0"]
        a1 = l["a1"]

        if ori == "h":
            if abs(c - top) <= 10 or abs(c - bottom) <= 10:
                continue
            x0 = max(a0, left)
            x1 = min(a1, right)
            if x1 - x0 < min_len:
                continue
            y0_mm = px_to_mm(bottom - (c + half_t), mm_per_px)
            y1_mm = px_to_mm(bottom - (c - half_t), mm_per_px)
            fp = {
                "min_x": px_to_mm(x0 - left, mm_per_px),
                "max_x": px_to_mm(x1 - left, mm_per_px),
                "min_y": min(y0_mm, y1_mm),
                "max_y": max(y0_mm, y1_mm),
            }
        else:
            if abs(c - left) <= 10 or abs(c - right) <= 10:
                continue
            y0 = max(a0, top)
            y1 = min(a1, bottom)
            if y1 - y0 < min_len:
                continue
            x0_mm = px_to_mm((c - half_t) - left, mm_per_px)
            x1_mm = px_to_mm((c + half_t) - left, mm_per_px)
            fp = {
                "min_x": min(x0_mm, x1_mm),
                "max_x": max(x0_mm, x1_mm),
                "min_y": px_to_mm(bottom - y1, mm_per_px),
                "max_y": px_to_mm(bottom - y0, mm_per_px),
            }

        if fp["max_x"] <= fp["min_x"] or fp["max_y"] <= fp["min_y"]:
            continue

        duplicate = False
        for ex in inner:
            exf = ex["footprint"]
            if (
                abs(exf["min_x"] - fp["min_x"]) <= 40
                and abs(exf["max_x"] - fp["max_x"]) <= 40
                and abs(exf["min_y"] - fp["min_y"]) <= 40
                and abs(exf["max_y"] - fp["max_y"]) <= 40
            ):
                duplicate = True
                break
        if duplicate:
            continue

        inner.append({"id": f"IW{idx}", "footprint": fp, "height": 3000})
        idx += 1

    return inner


def detect_colored_openings(img_bgr: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, np.array([0, 80, 40]), np.array([12, 255, 255]))
    red2 = cv2.inRange(hsv, np.array([170, 80, 40]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red1, red2)
    blue_mask = cv2.inRange(hsv, np.array([95, 70, 40]), np.array([130, 255, 255]))

    k = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, k)
    blue_mask = cv2.morphologyEx(blue_mask, cv2.MORPH_OPEN, k)

    if int(np.count_nonzero(red_mask)) + int(np.count_nonzero(blue_mask)) < 300:
        return [], []

    cnt_red, _ = cv2.findContours(red_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnt_blue, _ = cv2.findContours(blue_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [c for c in cnt_red if cv2.contourArea(c) > 80], [c for c in cnt_blue if cv2.contourArea(c) > 80]


def contour_passes_opening_shape_filter(contour: np.ndarray, wall_id: str, cx: float, cy: float, left: float, right: float, top: float, bottom: float) -> bool:
    rect = cv2.minAreaRect(contour)
    w, h = rect[1]
    if w <= 1 or h <= 1:
        return False
    long_side = max(w, h)
    short_side = min(w, h)
    if short_side > 26 or long_side < 8:
        return False

    wall_band = 18
    if wall_id == "W1":
        if abs(cx - left) > wall_band or h < (w * 1.4):
            return False
    elif wall_id == "W3":
        if abs(cx - right) > wall_band or h < (w * 1.4):
            return False
    elif wall_id == "W2":
        if abs(cy - bottom) > wall_band or w < (h * 1.4):
            return False
    else:
        if abs(cy - top) > wall_band or w < (h * 1.4):
            return False

    return True


def assign_wall(cx: float, cy: float, left: float, right: float, top: float, bottom: float) -> str:
    d = {"W1": abs(cx - left), "W2": abs(cy - bottom), "W3": abs(cx - right), "W4": abs(cy - top)}
    return min(d, key=d.get)


def contour_to_opening(contour: np.ndarray, opening_type: str, wall_id: str, left: float, right: float, top: float, bottom: float, mm_per_px: float, room_len_mm: int, room_wid_mm: int) -> dict[str, Any]:
    x, y, w, h = cv2.boundingRect(contour)
    x0, x1 = float(x), float(x + w)
    y0, y1 = float(y), float(y + h)

    if wall_id in ("W2", "W4"):
        start_mm = (x0 - left) * mm_per_px
        end_mm = (x1 - left) * mm_per_px
        wall_len = room_len_mm
    else:
        start_mm = (bottom - y1) * mm_per_px
        end_mm = (bottom - y0) * mm_per_px
        wall_len = room_wid_mm

    off = clamp_int(start_mm, 0, wall_len)
    end = clamp_int(end_mm, 0, wall_len)
    width_mm = max(1, end - off)

    out = {
        "type": opening_type,
        "wall_id": wall_id,
        "offset_from_start": off,
        "width": width_mm,
        "height": DEFAULT_DOOR_HEIGHT_MM if opening_type == "door" else DEFAULT_WINDOW_HEIGHT_MM,
    }
    if opening_type == "window":
        out["sill_height"] = DEFAULT_WINDOW_SILL_MM
    return out


def build_wall_defs(length_mm: int, width_mm: int) -> list[dict[str, Any]]:
    t = DEFAULT_WALL_THICKNESS_MM
    return [
        {"id": "W1", "start": [0, 0], "end": [0, width_mm], "thickness": t, "inward_normal": [1, 0]},
        {"id": "W2", "start": [0, 0], "end": [length_mm, 0], "thickness": t, "inward_normal": [0, 1]},
        {"id": "W3", "start": [length_mm, 0], "end": [length_mm, width_mm], "thickness": t, "inward_normal": [-1, 0]},
        {"id": "W4", "start": [0, width_mm], "end": [length_mm, width_mm], "thickness": t, "inward_normal": [0, -1]},
    ]


def detect_object_edges(
    gray: np.ndarray,
    left: float,
    right: float,
    top: float,
    bottom: float,
    mm_per_px: float,
    tokens: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    # Strong binary image for symbol edges.
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Remove very thin wall lines from dominating results.
    k = np.ones((2, 2), np.uint8)
    clean = cv2.morphologyEx(thr, cv2.MORPH_OPEN, k)

    contours, _ = cv2.findContours(clean, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    room_w = max(1.0, right - left)
    room_h = max(1.0, bottom - top)
    min_area = max(40.0, 0.00008 * room_w * room_h)
    max_area = 0.08 * room_w * room_h

    objs: list[dict[str, Any]] = []
    text_boxes = []
    for t in tokens:
        tx0 = int(t.get("x", 0))
        ty0 = int(t.get("y", 0))
        tx1 = tx0 + int(t.get("w", 0))
        ty1 = ty0 + int(t.get("h", 0))
        text_boxes.append((tx0, ty0, tx1, ty1))

    idx = 1
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = float(w * h)
        if area < min_area or area > max_area:
            continue

        # Keep objects mostly inside room interior.
        if x < left + 8 or y < top + 8 or (x + w) > right - 8 or (y + h) > bottom - 8:
            continue

        # Drop long wall-like segments.
        ar = max(w / max(1.0, h), h / max(1.0, w))
        if ar > 8.0:
            continue

        # Drop contours mostly overlapping OCR text (labels/dimensions).
        bb = (int(x), int(y), int(x + w), int(y + h))
        if any(_bbox_iou_like(bb, tb) > 0.25 for tb in text_boxes):
            continue

        fp = {
            "min_x": px_to_mm(x - left, mm_per_px),
            "max_x": px_to_mm((x + w) - left, mm_per_px),
            "min_y": px_to_mm(bottom - (y + h), mm_per_px),
            "max_y": px_to_mm(bottom - y, mm_per_px),
        }
        if fp["max_x"] <= fp["min_x"] or fp["max_y"] <= fp["min_y"]:
            continue

        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        label = None
        label_raw = None
        best_d = 1e9
        for t in tokens:
            up = re.sub(r"[^A-Z]", "", t["text"].upper())
            if not up:
                continue
            mapped = None
            for kword, ktype in FIXTURE_KEYWORDS.items():
                if kword in up:
                    mapped = ktype
                    break
            if mapped is None:
                continue
            d = float(np.hypot(t["cx"] - cx, t["cy"] - cy))
            if d < best_d and d <= 80:
                best_d = d
                label = mapped
                label_raw = t["text"]

        obj = {
            "id": f"OBJ{idx}",
            "type": label or "unknown_object",
            "source": "edge_contour",
            "footprint": fp,
            "bbox_px": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)},
        }
        if label_raw:
            obj["label_text"] = label_raw

        # de-dup near-identical boxes
        dup = False
        for ex in objs:
            ef = ex["footprint"]
            if (
                abs(ef["min_x"] - fp["min_x"]) <= 20
                and abs(ef["max_x"] - fp["max_x"]) <= 20
                and abs(ef["min_y"] - fp["min_y"]) <= 20
                and abs(ef["max_y"] - fp["max_y"]) <= 20
            ):
                dup = True
                break
        if dup:
            continue

        objs.append(obj)
        idx += 1

    return objs


def _safe_json_load(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    raw = text.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    i = raw.find("{")
    j = raw.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(raw[i : j + 1])
        except Exception:
            return None
    return None


def refine_state_with_dspy(
    state: dict[str, Any],
    gemini_api_key: str,
    model_name: str = "gemini/gemini-2.5-flash",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """
    AI refinement (DSPy + Gemini):
    - relabel rooms
    - relabel unknown objects into fixture categories
    - drop obvious noise objects
    Geometry is preserved.
    """
    import dspy

    lm = dspy.LM(
        model_name,
        api_key=gemini_api_key,
        temperature=0.0,
        max_tokens=2000,
    )
    dspy.configure(lm=lm)
    predictor = dspy.Predict("draft_json, task_rules -> refinement_json")

    draft = {
        "rooms": state.get("rooms", []),
        "objects": state.get("objects", []),
        "text_annotations": state.get("text_annotations", [])[:120],
        "space": state.get("space", {}),
        "inner_walls": state.get("inner_walls", []),
    }
    task_rules = {
        "rules": [
            "Only use existing ids from draft_json.",
            "Do not invent geometry or new ids.",
            "Fix room names and object types only.",
            "Use fixture types from: sink, wash_basin, wash_area, stove, gas_stove, hob, cooktop, wc, toilet_fixture, shower, fridge, chimney, unknown_object.",
            "Return strict JSON.",
        ],
        "output_schema": {
            "room_updates": [{"id": 1, "name": "BEDROOM", "label_text": "BEDROOM"}],
            "object_updates": [{"id": "OBJ1", "type": "sink"}],
            "drop_object_ids": ["OBJ9"],
        },
    }

    pred = predictor(
        draft_json=json.dumps(draft, ensure_ascii=False),
        task_rules=json.dumps(task_rules, ensure_ascii=False),
    )

    raw = getattr(pred, "refinement_json", None) or str(pred)
    refinement = _safe_json_load(raw)
    if refinement is None:
        raise ValueError("DSPy refinement was not valid JSON")

    room_updates = {
        int(x["id"]): x
        for x in refinement.get("room_updates", [])
        if isinstance(x, dict) and x.get("id") is not None
    }
    obj_updates = {
        str(x["id"]): x
        for x in refinement.get("object_updates", [])
        if isinstance(x, dict) and x.get("id")
    }
    drop_ids = {str(x) for x in refinement.get("drop_object_ids", [])}

    for room in state.get("rooms", []):
        rid = int(room.get("id", -1))
        upd = room_updates.get(rid)
        if not upd:
            continue
        if isinstance(upd.get("name"), str) and upd["name"].strip():
            room["name"] = upd["name"].strip()
        if isinstance(upd.get("label_text"), str) and upd["label_text"].strip():
            room["label_text"] = upd["label_text"].strip()

    new_objects = []
    for obj in state.get("objects", []):
        oid = str(obj.get("id", ""))
        if oid in drop_ids:
            continue
        upd = obj_updates.get(oid)
        if upd and isinstance(upd.get("type"), str) and upd["type"].strip():
            obj["type"] = upd["type"].strip()
            obj["source"] = str(obj.get("source", "edge_contour")) + "+ai"
        new_objects.append(obj)
    state["objects"] = new_objects

    state["services"] = {
        "points": [
            {
                "id": obj["id"],
                "service_type": obj.get("type", "unknown_object"),
                "footprint": obj.get("footprint", {}),
            }
            for obj in state.get("objects", [])
        ]
    }

    return state, refinement


def classify_unknown_objects_with_vision(
    state: dict[str, Any],
    image_path: Path,
    gemini_api_key: str,
    model_name: str = "gemini-2.0-flash",
    max_objects: int = 0,
) -> dict[str, Any]:
    """
    Force AI vision pass on unknown objects.
    Updates object.type in-place for unknown_object entries based on image crops.
    """
    from google import genai
    from google.genai import types

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image for AI object classification: {image_path}")

    client = genai.Client(api_key=gemini_api_key)
    allowed = [
        "sink",
        "wash_basin",
        "wash_area",
        "stove",
        "gas_stove",
        "hob",
        "cooktop",
        "wc",
        "toilet_fixture",
        "shower",
        "fridge",
        "chimney",
        "door_symbol",
        "window_symbol",
        "unknown_object",
    ]

    unknown = [o for o in state.get("objects", []) if o.get("type") == "unknown_object" and o.get("bbox_px")]
    unknown.sort(
        key=lambda o: int(o.get("bbox_px", {}).get("w", 0)) * int(o.get("bbox_px", {}).get("h", 0)),
        reverse=True,
    )
    if max_objects and max_objects > 0:
        unknown = unknown[:max_objects]

    updated = 0
    attempted = 0
    for obj in unknown:
        bb = obj.get("bbox_px", {})
        x = int(bb.get("x", 0))
        y = int(bb.get("y", 0))
        w = int(bb.get("w", 0))
        h = int(bb.get("h", 0))
        if w <= 1 or h <= 1:
            continue

        margin = 18
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(img.shape[1], x + w + margin)
        y1 = min(img.shape[0], y + h + margin)
        crop = img[y0:y1, x0:x1]
        if crop.size == 0:
            continue

        ok, enc = cv2.imencode(".png", crop)
        if not ok:
            continue
        attempted += 1

        prompt = (
            "Classify this CAD floor-plan symbol crop. "
            f"Return ONLY JSON: {{\"type\":\"one_of_{allowed}\",\"confidence\":0_to_1,\"reason\":\"short\"}}. "
            "If uncertain, return unknown_object."
        )

        try:
            response = client.models.generate_content(
                model=model_name,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_bytes(data=enc.tobytes(), mime_type="image/png"),
                            types.Part.from_text(text=prompt),
                        ],
                    )
                ],
                config=types.GenerateContentConfig(temperature=0.0),
            )
            raw = (response.text or "").strip()
            parsed = _safe_json_load(raw)
            if not parsed:
                continue
            typ = str(parsed.get("type", "")).strip()
            if typ not in allowed:
                continue
            obj["type"] = typ
            obj["ai_confidence"] = float(parsed.get("confidence", 0.0))
            if parsed.get("reason"):
                obj["ai_reason"] = str(parsed.get("reason"))
            obj["source"] = str(obj.get("source", "edge_contour")) + "+vision_ai"
            if typ != "unknown_object":
                updated += 1
        except Exception:
            continue

    # Keep services in sync with object types.
    state["services"] = {
        "points": [
            {
                "id": obj["id"],
                "service_type": obj.get("type", "unknown_object"),
                "footprint": obj.get("footprint", {}),
            }
            for obj in state.get("objects", [])
        ]
    }
    return {"attempted": attempted, "updated": updated, "total_unknown_considered": len(unknown)}


def refine_room_dimensions_with_vision(
    state: dict[str, Any],
    image_path: Path,
    gemini_api_key: str,
    model_name: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """
    Use Gemini vision to read actual dimension text from the floor plan image
    and overwrite dimensions_ft for each matched room in state.
    Returns stats dict: {"attempted": int, "updated": int}.
    """
    from google import genai
    from google.genai import types

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image for Gemini request")

    client = genai.Client(api_key=gemini_api_key)

    rooms = state.get("rooms", [])
    room_names = [r.get("name", "") for r in rooms if r.get("name")]
    if not room_names:
        return {"attempted": 0, "updated": 0}

    names_list = ", ".join(room_names)
    prompt = (
        f"This is a floor plan image. The following rooms have been detected: {names_list}.\n"
        "For each room, find and read the dimension text printed inside the room boundary. "
        "Dimensions may appear in formats like: '9.4 x 7.3', '9'4\" x 7'3\"', '11.6x8.3', etc.\n"
        "Return ONLY valid JSON in this exact format:\n"
        '{"rooms": [{"name": "<room_name>", "length_ft": <float>, "width_ft": <float>}]}\n'
        "Rules:\n"
        "- name must exactly match one of the detected rooms listed above\n"
        "- Convert feet-inches to decimal feet (e.g. 9'4\" = 9.33)\n"
        "- Only include rooms where you can clearly read dimension text\n"
        "- Do not guess or hallucinate dimensions\n"
        "- If no dimensions are visible for a room, omit it from the list"
    )

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=[
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_bytes(data=enc.tobytes(), mime_type="image/png"),
                        types.Part.from_text(text=prompt),
                    ],
                )
            ],
            config=types.GenerateContentConfig(temperature=0.0),
        )
        raw = (response.text or "").strip()
        parsed = _safe_json_load(raw)
    except Exception:
        return {"attempted": 1, "updated": 0}

    if not parsed or not isinstance(parsed.get("rooms"), list):
        return {"attempted": 1, "updated": 0}

    # Build lookup by normalized room name
    room_by_name = {r.get("name", "").upper(): r for r in rooms}
    updated = 0
    for entry in parsed["rooms"]:
        name = str(entry.get("name", "")).upper()
        length = entry.get("length_ft")
        width = entry.get("width_ft")
        if name not in room_by_name:
            continue
        if length is None or width is None:
            continue
        try:
            length = float(length)
            width = float(width)
        except (TypeError, ValueError):
            continue
        if not (2.5 <= length <= 40 and 2.5 <= width <= 40):
            continue
        room_by_name[name]["dimensions_ft"] = {
            "length": round(length, 2),
            "width": round(width, 2),
        }
        room_by_name[name]["dimensions_source"] = "vision_ai"
        updated += 1

    return {"attempted": 1, "updated": updated}


def _to_ints_deep(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_ints_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_ints_deep(v) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, str):
        return obj
    if isinstance(obj, (int, float)):
        return int(round(float(obj)))
    return obj


def extract_json_with_user_prompt(
    image_path: Path,
    gemini_api_key: str,
    target_room: str | None = None,
    model_name: str = "gemini-2.0-flash",
) -> dict[str, Any]:
    """
    Uses the user's strict prompt template to return final JSON directly from image.
    """
    from google import genai
    from google.genai import types

    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image for Gemini request")

    client = genai.Client(api_key=gemini_api_key)

    room_line = (
        f'Identify the specified room: "{target_room}".'
        if target_room
        else "Identify the specified room from context in the image."
    )

    prompt = f"""
You are a building plan interpretation engine.

Input:
A floor plan image.
The image contains labeled rooms with dimensions written in feet and inches (for example: 8'x10', 11'x14', 5'x7'6", etc.).

Your task:
{room_line}
Read the room dimensions from the label inside the room.
Convert all dimensions from feet/inches to millimeters.
1 foot = 304.8 mm
1 inch = 25.4 mm

Assume:
Rectangular footprint.
Ceiling height = 3000 mm (unless otherwise specified in the image).
Wall thickness = 150 mm.

Generate structured JSON strictly in the following format.
If information such as openings, columns, services, beams, or inner walls is not clearly visible, return empty arrays for them.
Do not hallucinate data. Only extract what is visible in the image.
Output must be valid JSON only. No explanations.

JSON format:
{{
  "units": "mm",
  "space": {{
    "id": 1,
    "name": "<Room Name>",
    "type": "<room_type_lowercase>",
    "height": 3000,
    "footprint": {{
      "min_x": 0,
      "min_y": 0,
      "max_x": <converted_width_mm>,
      "max_y": <converted_length_mm>
    }}
  }},
  "exterior_walls": [
    {{
      "id": "W1",
      "label": "A",
      "start": [0, 0],
      "end": [0, <length_mm>],
      "thickness": 150,
      "inward_normal": [1, 0]
    }},
    {{
      "id": "W2",
      "label": "B",
      "start": [0, 0],
      "end": [<width_mm>, 0],
      "thickness": 150,
      "inward_normal": [0, 1]
    }},
    {{
      "id": "W3",
      "label": "C",
      "start": [<width_mm>, 0],
      "end": [<width_mm>, <length_mm>],
      "thickness": 150,
      "inward_normal": [-1, 0]
    }},
    {{
      "id": "W4",
      "label": "D",
      "start": [0, <length_mm>],
      "end": [<width_mm>, <length_mm>],
      "thickness": 150,
      "inward_normal": [0, -1]
    }}
  ],
  "openings": [],
  "columns": [],
  "inner_walls": [],
  "beams": [],
  "services": {{
    "points": []
  }}
}}

Constraints:
All numeric values must be integers (round to nearest mm).
Preserve exact proportions from the image labels.
Do not invent dimensions.
"""

    response = client.models.generate_content(
        model=model_name,
        contents=[
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=enc.tobytes(), mime_type="image/png"),
                    types.Part.from_text(text=prompt),
                ],
            )
        ],
        config=types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"),
    )

    parsed = _safe_json_load(response.text or "")
    if parsed is None:
        raise ValueError("Prompt JSON mode: model did not return valid JSON")
    parsed = _to_ints_deep(parsed)
    return parsed


def extract_floorplan(
    image_path: Path,
    mm_per_px: float | None,
    tessdata_dir: str | None,
    auto_scale: bool = False,
    ocr_engine: str = "paddle",
) -> dict[str, Any]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    merged_lines = detect_lines(gray)
    left, right, top, bottom = detect_outer_frame(merged_lines, w, h)

    components = detect_room_components(gray)
    tokens = ocr_tokens(img, tessdata_dir=tessdata_dir, engine=ocr_engine)

    # First pass with default/fixed scale for provisional room footprints.
    base_scale = mm_per_px if mm_per_px is not None else DEFAULT_MM_PER_PX
    provisional_rooms = build_labeled_rooms(tokens, components, base_scale)

    # Optional auto-calibration from OCR dimensions.
    if mm_per_px is None and auto_scale:
        mm_per_px = calibrate_mm_per_px_from_rooms(provisional_rooms, components, base_scale)
    elif mm_per_px is None:
        mm_per_px = base_scale

    room_len_mm = max(1, px_to_mm(right - left, mm_per_px))
    room_wid_mm = max(1, px_to_mm(bottom - top, mm_per_px))

    # Rebuild rooms with final scale. Combine both detectors to reduce misses.
    rooms_comp = build_labeled_rooms(tokens, components, mm_per_px)
    rooms_line = build_labeled_rooms_from_lines(tokens, merged_lines, left, right, top, bottom, mm_per_px)
    rooms = merge_room_sets(rooms_comp, rooms_line) if rooms_comp else rooms_line
    clamp_rooms_to_space(rooms, room_len_mm, room_wid_mm)
    rooms = filter_invalid_rooms(rooms)

    inner_walls = build_inner_walls(merged_lines, left, right, top, bottom, mm_per_px)
    objects = detect_object_edges(gray, left, right, top, bottom, mm_per_px, tokens)

    door_cnts, win_cnts = detect_colored_openings(img)
    openings: list[dict[str, Any]] = []
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
            "ocr_engine": ocr_engine,
        },
    }
    return state


def opening_segment(state: dict[str, Any], opening: dict[str, Any]) -> tuple[tuple[float, float], tuple[float, float]]:
    wall = next(w for w in state["exterior_walls"] if w["id"] == opening["wall_id"])
    x1, y1 = wall["start"]
    x2, y2 = wall["end"]
    dx = x2 - x1
    dy = y2 - y1
    length = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / length, dy / length
    off = opening["offset_from_start"]
    wid = opening["width"]
    sx = x1 + ux * off
    sy = y1 + uy * off
    ex = sx + ux * wid
    ey = sy + uy * wid
    return (sx, sy), (ex, ey)


def write_svg(state: dict[str, Any], out_svg: Path) -> None:
    fp = state["space"]["footprint"]
    max_x = fp["max_x"]
    max_y = fp["max_y"]
    pad = 220
    vb_x = -pad
    vb_y = -(max_y + pad)
    vb_w = max_x + 2 * pad
    vb_h = max_y + 2 * pad

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="{vb_x} {vb_y} {vb_w} {vb_h}">')
    lines.append('<rect x="-100000" y="-100000" width="200000" height="200000" fill="white"/>')
    lines.append('<g transform="scale(1,-1)">')

    for w in state["exterior_walls"]:
        x1, y1 = w["start"]
        x2, y2 = w["end"]
        lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#111" stroke-width="30" stroke-linecap="square"/>')

    for iw in state.get("inner_walls", []):
        fpw = iw.get("footprint", {})
        x = fpw.get("min_x", 0)
        y = fpw.get("min_y", 0)
        wv = max(1, fpw.get("max_x", 0) - fpw.get("min_x", 0))
        hv = max(1, fpw.get("max_y", 0) - fpw.get("min_y", 0))
        lines.append(f'<rect x="{x}" y="{y}" width="{wv}" height="{hv}" fill="#bfbfbf" stroke="#555" stroke-width="4"/>')

    for o in state["openings"]:
        (sx, sy), (ex, ey) = opening_segment(state, o)
        color = "#cc4b37" if o["type"] == "door" else "#0b84f3"
        lines.append(f'<line x1="{sx}" y1="{sy}" x2="{ex}" y2="{ey}" stroke="{color}" stroke-width="36" stroke-linecap="butt"/>')

    for r in state.get("rooms", []):
        rfp = r.get("footprint", {})
        cx = (rfp.get("min_x", 0) + rfp.get("max_x", 0)) / 2.0
        cy = (rfp.get("min_y", 0) + rfp.get("max_y", 0)) / 2.0
        label = r.get("label_text", r.get("name", "Room"))
        lines.append(
            f'<text x="{cx}" y="{cy}" fill="#222" font-size="56" text-anchor="middle" '
            f'font-family="sans-serif" transform="scale(1,-1) translate(0,{-2 * cy})">{label}</text>'
        )

    for obj in state.get("objects", []):
        fp = obj.get("footprint", {})
        x = fp.get("min_x", 0)
        y = fp.get("min_y", 0)
        w = max(1, fp.get("max_x", 0) - fp.get("min_x", 0))
        h = max(1, fp.get("max_y", 0) - fp.get("min_y", 0))
        lines.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="none" stroke="#0a9d58" stroke-width="6"/>'
        )
        tlabel = obj.get("type", "obj")
        cx = x + (w / 2.0)
        if tlabel == "unknown_object":
            cy = y + (h / 2.0)
            lines.append(
                f'<text x="{cx}" y="{cy}" fill="#0a9d58" font-size="42" text-anchor="middle" '
                f'font-family="sans-serif" transform="scale(1,-1) translate(0,{-2 * cy})">?</text>'
            )
        else:
            cy = y + h + 25
            lines.append(
                f'<text x="{cx}" y="{cy}" fill="#0a9d58" font-size="32" text-anchor="middle" '
                f'font-family="sans-serif" transform="scale(1,-1) translate(0,{-2 * cy})">{tlabel}</text>'
            )

    lines.append('</g>')
    lines.append('</svg>')
    out_svg.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="CAD map image parser: image -> floorplan_state.json + final.svg")
    parser.add_argument("--image", required=True, help="Input CAD/floor-plan image path")
    parser.add_argument("--json-out", default="floorplan_state.json", help="Output JSON file")
    parser.add_argument("--svg-out", default="final.svg", help="Output SVG file")
    parser.add_argument("--mm-per-px", type=float, default=None, help="Scale override (mm per pixel)")
    parser.add_argument("--auto-scale", action="store_true", help="Enable OCR-based mm/px auto calibration")
    parser.add_argument("--tessdata-dir", default=None, help="Custom tessdata directory for OCR")
    parser.add_argument("--ocr-engine", default="paddle", choices=["paddle", "easyocr", "tesseract"], help="OCR engine for text extraction")
    parser.add_argument("--use-ai", action="store_true", help="Use DSPy + Gemini refinement")
    parser.add_argument("--gemini-model", default="gemini/gemini-2.5-flash", help="DSPy model name")
    parser.add_argument("--force-ai-objects", action="store_true", help="Force Gemini vision on unknown objects")
    parser.add_argument("--vision-model", default="gemini-2.0-flash", help="Gemini vision model")
    parser.add_argument("--ai-max-objects", type=int, default=0, help="Max unknown objects for vision pass (0=all)")
    parser.add_argument("--ai-dimensions", action="store_true", help="Use Gemini vision to read and correct room dimensions from the image")
    parser.add_argument("--prompt-json", action="store_true", help="Use strict prompt-only JSON extraction mode")
    parser.add_argument("--target-room", default=None, help="Room name to target in prompt JSON mode (e.g. Kitchen)")
    args = parser.parse_args()

    if args.prompt_json:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before running with --prompt-json.")
        state = extract_json_with_user_prompt(
            image_path=Path(args.image),
            gemini_api_key=key,
            target_room=args.target_room,
            model_name=args.vision_model,
        )
        # Ensure a minimal meta block for logging/traceability.
        if isinstance(state, dict):
            state.setdefault("meta", {})
            state["meta"]["prompt_json_mode"] = {
                "enabled": True,
                "model": args.vision_model,
                "target_room": args.target_room,
            }
    else:
        state = extract_floorplan(
            Path(args.image),
            mm_per_px=args.mm_per_px,
            tessdata_dir=args.tessdata_dir,
            auto_scale=args.auto_scale,
            ocr_engine=args.ocr_engine,
        )

    ai_meta: dict[str, Any] = {"enabled": False}
    if args.use_ai:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before running with --use-ai.")
        state, ai_raw = refine_state_with_dspy(state, gemini_api_key=key, model_name=args.gemini_model)
        ai_meta = {"enabled": True, "model": args.gemini_model, "raw_refinement": ai_raw}

    if args.force_ai_objects:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before running with --force-ai-objects.")
        vision_stats = classify_unknown_objects_with_vision(
            state=state,
            image_path=Path(args.image),
            gemini_api_key=key,
            model_name=args.vision_model,
            max_objects=args.ai_max_objects,
        )
        ai_meta["vision_object_refinement"] = {
            "enabled": True,
            "model": args.vision_model,
            **vision_stats,
        }

    if args.ai_dimensions:
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("GEMINI_API_KEY is not set. Export it before running with --ai-dimensions.")
        dim_stats = refine_room_dimensions_with_vision(
            state=state,
            image_path=Path(args.image),
            gemini_api_key=key,
            model_name=args.vision_model,
        )
        ai_meta["dimension_refinement"] = {
            "enabled": True,
            "model": args.vision_model,
            **dim_stats,
        }

    state.setdefault("meta", {})
    state["meta"]["ai_refinement"] = ai_meta

    Path(args.json_out).write_text(json.dumps(state, indent=2))
    write_svg(state, Path(args.svg_out))

    print(f"Wrote {args.json_out}")
    print(f"Wrote {args.svg_out}")
    if not args.prompt_json:
        print(
            f"Rooms: {len(state.get('rooms', []))}, Inner walls: {len(state.get('inner_walls', []))}, "
            f"Openings: {len(state.get('openings', []))}, Objects: {len(state.get('objects', []))}"
        )
        print(f"Scale: {state.get('meta', {}).get('scale_mm_per_px')} mm/px")
        print(f"OCR Engine: {state.get('meta', {}).get('ocr_engine')}")
        print("Detected key room dimensions:")
        printed_any = False
        for room in state.get("rooms", []):
            name = str(room.get("name", ""))
            up = name.upper()
            if not any(k in up for k in ("KITCHEN", "BEDROOM", "TOILET")):
                continue
            dims_ft = room.get("dimensions_ft")
            fp = room.get("footprint", {})
            w_mm = int(fp.get("max_x", 0) - fp.get("min_x", 0))
            h_mm = int(fp.get("max_y", 0) - fp.get("min_y", 0))
            if dims_ft:
                print(
                    f"- {name}: {dims_ft.get('length')}ft x {dims_ft.get('width')}ft "
                    f"(bbox: {w_mm}mm x {h_mm}mm)"
                )
            else:
                print(f"- {name}: dimension text not found (bbox: {w_mm}mm x {h_mm}mm)")
            printed_any = True
        if not printed_any:
            print("- No Kitchen/Bedroom/Toilet labels detected.")
        if args.use_ai:
            print(f"AI refinement: enabled ({args.gemini_model})")
        if args.force_ai_objects:
            v = state.get("meta", {}).get("ai_refinement", {}).get("vision_object_refinement", {})
            print(
                f"AI object vision: enabled ({args.vision_model}), "
                f"attempted={v.get('attempted', 0)}, updated={v.get('updated', 0)}"
            )
        if args.ai_dimensions:
            d = state.get("meta", {}).get("ai_refinement", {}).get("dimension_refinement", {})
            print(
                f"AI dimension vision: enabled ({args.vision_model}), "
                f"updated={d.get('updated', 0)} rooms"
            )
    else:
        print(f"Prompt JSON mode: enabled ({args.vision_model})")


if __name__ == "__main__":
    main()
