import argparse
import re

import cv2
import pytesseract

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

    return best_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Fast, higher-accuracy OCR for floor-plan text")
    parser.add_argument("--image", default="C1.jpg", help="Input image path")
    args = parser.parse_args()

    text = extract_best_text(args.image)
    print("Extracted Text:\n", text)


if __name__ == "__main__":
    main()
