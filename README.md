# Floor Extractor

Floor Extractor converts floor-plan images into structured JSON and an SVG preview.

It combines OpenCV geometry detection with OCR, and supports optional cloud AI extractors.

## What it does

- Detects space boundary, walls, inner walls, and openings
- Extracts room labels and visible dimensions from image text
- Detects object/fixture-like symbols
- Writes normalized state JSON and rendered SVG
- Supports optional Google Vision and Gemini-based extraction flows

## Repository layout

- `extract_floor_plan.py`: main extractor (OpenCV + local OCR)
- `extract_floor_plan_google_vision.py`: extractor using Google Cloud Vision OCR
- `extract_floor_plan_gemini.py`: extractor using Gemini vision model
- `backend_functions.py`: helper CLI for creating/updating/rendering plan data
- `requirements.txt`: Python dependencies for `pip install -r`
- `tessdata/`: local Tesseract language data

## Requirements

- Python 3.10+
- Tesseract OCR installed on system (for local OCR mode)
- Python dependencies from `requirements.txt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install Tesseract:

- Ubuntu/Debian: `sudo apt-get install -y tesseract-ocr`
- Arch: `sudo pacman -S tesseract`
- macOS (Homebrew): `brew install tesseract`

## Basic usage

```bash
python extract_floor_plan.py \
  --image ./floor2.png \
  --json-out floorplan_state.json \
  --svg-out final.svg \
  --tessdata-dir ./tessdata
```

## Google Vision usage

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/service-account.json"
python extract_floor_plan_google_vision.py \
  --image ./floor2.png \
  --json-out floorplan_state.json \
  --svg-out final.svg
```

## Gemini usage

```bash
export GEMINI_API_KEY="your_api_key"
python extract_floor_plan_gemini.py \
  --image ./floor2.png \
  --json-out floorplan_state.json \
  --svg-out final.svg
```

## Validate setup

```bash
python validate_setup.py
```

## Notes

- Local OCR quality depends heavily on text clarity and scale in the source image.
- For `extract_floor_plan.py`, ensure `--tessdata-dir` points to valid language files.
- Cloud extractor scripts require credentials and corresponding packages installed.
