# CLAUDE.md

This file guides Claude Code for this project.

## Project Goal

Parse floor-plan images into:
- `floorplan_state.json`
- `final.svg`

Primary script is expected to be `extract_floor_plan.py`.

## Environment

- Python 3.10+
- Use a virtual environment

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies (minimum):

```bash
pip install opencv-python numpy pytesseract
```

Optional better OCR + AI:

```bash
pip install paddleocr easyocr google-genai dspy
```

## Run

Basic:

```bash
py extract_floor_plan.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg \
  --tessdata-dir ./tessdata \
  --ocr-engine paddle
```

Tesseract fallback:

```bash
py extract_floor_plan.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg \
  --tessdata-dir ./tessdata \
  --ocr-engine tesseract
```

Optional AI object classification:

```bash
export GEMINI_API_KEY="<your_key>"
py extract_floor_plan.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg \
  --tessdata-dir ./tessdata \
  --ocr-engine paddle \
  --force-ai-objects \
  --vision-model gemini-2.0-flash
```

## Notes for Claude Code

- If output shows `Rooms: 0`, this is usually OCR path/engine failure.
- Verify `tessdata/eng.traineddata` exists.
- `--force-ai-objects` labels object boxes only; it does not recover missing room labels.
- Prefer improving OCR + dimension regex + room-label linking before adding more AI logic.

## Debug Checklist

1. Confirm image path and quality.
2. Confirm OCR assets:
   - `ls -la ./tessdata`
3. Try OCR engine switching:
   - `--ocr-engine paddle`
   - `--ocr-engine easyocr`
   - `--ocr-engine tesseract`
4. Run without AI flags first.
5. Inspect `text_annotations` in JSON for detected OCR tokens.

## Code Change Rules

- Keep JSON schema stable.
- Keep units in `mm`.
- Avoid removing CLI flags unless explicitly requested.
- Prefer small focused patches.

## Validation

```bash
python3 -m py_compile extract_floor_plan.py
py extract_floor_plan.py --image ./floor5.jpg --json-out floorplan_state.json --svg-out final.svg --tessdata-dir ./tessdata --ocr-engine paddle
```
