import os
import re

import cv2
import pytesseract
from PIL import Image

from utils import organize_floor_data


ROOM_HINTS = [
    "KITCHEN",
    "BEDROOM",
    "LIVING",
    "BALCONY",
    "CLOSET",
    "BATH",
    "LAUNDRY",
    "DINING",
    "TOILET",
]

DIM_PAT = re.compile(r"\d{1,2}\s*['\"]?\s*(?:x|X|×)\s*\d{1,2}", re.IGNORECASE)
AREA_PAT = re.compile(r"\d{1,4}(?:\.\d+)?\s*(?:SQ\.?\s*FT|SQFT|SFT|FT2|FT\^2)", re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = text.replace("×", "x").replace("—", "-")
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return text


def ocr_score(text: str) -> int:
    up = text.upper()
    dim_count = len(DIM_PAT.findall(up))
    area_count = len(AREA_PAT.findall(up))
    room_count = sum(1 for keyword in ROOM_HINTS if keyword in up)
    digit_count = sum(ch.isdigit() for ch in up)
    return dim_count * 8 + area_count * 8 + room_count * 5 + min(digit_count, 30)


def ocr_with_config(img, psm: int) -> str:
    cfg = f"--oem 1 --psm {psm} -l eng"
    return pytesseract.image_to_string(img, config=cfg)


def build_variants(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    blur = cv2.GaussianBlur(up, (3, 3), 0)
    adap = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )
    otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    inv_otsu = cv2.bitwise_not(otsu)
    return [up, adap, otsu, inv_otsu]


def extract_best_text(image_path: str) -> str:
    print("**********1**********")
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    variants = build_variants(bgr)
    best_text = ""
    best_score = -1
    for variant in variants:
        for psm in (6, 11):
            text = normalize_text(ocr_with_config(variant, psm))
            score = ocr_score(text)
            if score > best_score:
                best_score = score
                best_text = text

    data = organize_floor_data(best_text)
    for room in data.get("rooms", []):
        if room.get("name") == "KITCHEN":
            print(room.get("raw_label"), room.get("dimensions") or room.get("area"))
    return best_text


def image_to_text(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        img = Image.open(image_path)
        print("**********2**********")
        text = pytesseract.image_to_string(img)
        data = organize_floor_data(text)
        for room in data.get("rooms", []):
            if room.get("name") == "KITCHEN":
                print(room.get("raw_label"), room.get("dimensions") or room.get("area"))
        return text.strip()
    except Exception as exc:
        raise RuntimeError(f"Error processing image: {exc}")
