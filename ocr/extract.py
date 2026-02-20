import argparse
import json
import os
import re
import sys

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


def normalize_text(t: str) -> str:
    t = t.replace("×", "x").replace("—", "-")
    t = t.replace("’", "'").replace("“", '"').replace("”", '"')
    return t


def ocr_score(text: str) -> int:
    up = text.upper()
    dim_count = len(DIM_PAT.findall(up))
    room_count = sum(1 for k in ROOM_HINTS if k in up)
    digit_count = sum(ch.isdigit() for ch in up)
    return dim_count * 8 + room_count * 5 + min(digit_count, 30)


def ocr_with_config(img, psm: int) -> str:
    cfg = f"--oem 1 --psm {psm} -l eng"
    return pytesseract.image_to_string(img, config=cfg)


def build_variants(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    up = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # Variant 1: light denoise + adaptive threshold
    blur = cv2.GaussianBlur(up, (3, 3), 0)
    adap = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        8,
    )

    # Variant 2: Otsu threshold
    otsu = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Variant 3: inverted Otsu (some drawings OCR better this way)
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

    # Keep it fast: 2 PSM modes across 4 variants.
    for v in variants:
        for psm in (6, 11):
            txt = normalize_text(ocr_with_config(v, psm))
            sc = ocr_score(txt)
            if sc > best_score:
                best_score = sc
                best_text = txt

    data = organize_floor_data(best_text)
    for room in data.get("rooms", []):
        if room.get("name") == "KITCHEN":
            name = room.get("name")
            label = room.get("raw_label")
            dims = room.get("dimensions")

            print(label, dims)

    return best_text


def image_to_text(image_path):
    """
    Extracts text from an image using Tesseract OCR.
    :param image_path: Path to the image file
    :return: Extracted text as a string
    """
    # Validate file existence
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"File not found: {image_path}")

    try:
        # Open the image
        img = Image.open(image_path)
        print("**********2**********")

        # Extract text
        text = pytesseract.image_to_string(img)
        data = organize_floor_data(text)

        # print(data["unit"])  # print unit
        # print(data["rooms"][1]["name"])  # print all rooms

        for room in data.get("rooms", []):
            if room.get("name") == "KITCHEN":
                name = room.get("name")
                label = room.get("raw_label")
                dims = room.get("dimensions")

                print(label, dims)

        return text.strip()

    except Exception as e:
        raise RuntimeError(f"Error processing image: {e}")


# text = extract_best_text(args.image)
# result = organize_floor_data(results["test_py_result"], results["test2_py_result"])
# extracted_text = image_to_text(image_path)
