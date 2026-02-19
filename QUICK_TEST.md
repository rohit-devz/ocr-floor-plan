# Quick Test Reference Card

## 🚀 Quick Start (Copy-Paste Ready)

```bash
# Step 1: Validate your setup (2 min)
python3 validate_setup.py

# Step 2: Compare methods on one image (2-5 min)
python3 test_method_compare.py

# Step 3: Full test suite if needed (5-60 min depending on images)
python3 test_all_extractors.py --quick
```

---

## 🎯 Choose Your Method

### If you see "✓ READY Local OCR" in validate_setup.py:
```bash
# Basic usage
python3 extract_floor_plan.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg \
  --tessdata-dir ./tessdata
```
⏱️ 30-40 seconds | 💰 Free | 📊 Good accuracy

---

### If you see "✓ AVAILABLE Gemini 2.5 Flash":
```bash
# First time setup (1 minute)
export GEMINI_API_KEY="your-api-key-from-aistudio.google.com"

# Then run
python3 extract_floor_plan_gemini.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg
```
⚡ <1 second | 💰 $0.000075/image | 📊 Excellent accuracy

---

### If you see "✓ AVAILABLE Google Cloud Vision":
```bash
# First time setup (complex, see GOOGLE_VISION_SETUP.md)
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"

# Then run
python3 extract_floor_plan_google_vision.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg
```
⏱️ 1-3 seconds | 💰 $0.0006/image | 📊 Excellent accuracy

---

## 📊 Test Commands Reference

| Task | Command | Time |
|------|---------|------|
| Check setup | `python3 validate_setup.py` | 1 min |
| Quick compare | `python3 test_method_compare.py` | 2-5 min |
| Full test | `python3 test_all_extractors.py` | 5-60 min |
| Parallel extract | `python3 test_extract_parallel.py` | 2 min |
| Benchmark OCR | `python3 benchmark_ocr.py` | 2-3 min |

---

## 💡 Which Test Should I Run?

```
START HERE
    ↓
    python3 validate_setup.py
    ↓
    See what methods are available?
    ↓
    ├─→ Want quick recommendation?
    │   └─→ python3 test_method_compare.py
    │       (Shows speed/cost/accuracy, 2-5 min)
    │
    └─→ Need production validation?
        └─→ python3 test_all_extractors.py --quick
            (Full test on first image, 5-10 min)

            Then:
            python3 test_all_extractors.py
            (Full test on all images, 30-60 min)
```

---

## 🔧 Setup Issues?

### Missing GEMINI_API_KEY?
```bash
# Get key from: https://aistudio.google.com/app/apikey
export GEMINI_API_KEY="sk_proj_..."
python3 validate_setup.py  # Should now show ✓
```

### Missing GOOGLE_APPLICATION_CREDENTIALS?
```bash
# See GOOGLE_VISION_SETUP.md for full setup
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/credentials.json"
python3 validate_setup.py  # Should now show ✓
```

### Missing tessdata?
```bash
# For local OCR, you need tessdata
ls tessdata/eng.traineddata  # Check if exists

# If missing, download from:
# https://github.com/UB-Mannheim/tesseract/wiki
```

---

## 📈 Performance at a Glance

For **1,000 floor plans** (worst case):

| Method | Time | Cost | Setup |
|--------|------|------|-------|
| **Gemini** ⭐ | 15 min | $0.23 | 5 min |
| Google Vision | 30 min | $0.60 | 30 min |
| Local OCR | 11 hours | $0.00 | 5 min |
| Local OCR (4x parallel) | 3 hours | $0.00 | 5 min |

**Gemini is 44x faster with excellent accuracy!**

---

## 📄 View Results

After running tests, check outputs:

```bash
# View extracted floor plan
cat test_results_compare/gemini/floor5.json | jq '.rooms'

# View visual preview
open test_results_compare/gemini/floor5.svg  # macOS
xdg-open test_results_compare/gemini/floor5.svg  # Linux
start test_results_compare/gemini/floor5.svg  # Windows
```

---

## 🏃 One-Liner: Test Everything

```bash
python3 validate_setup.py && python3 test_method_compare.py && echo "✓ Done"
```

---

## 📚 Full Documentation

- **Setup Guide**: See `GEMINI_SETUP.md` or `GOOGLE_VISION_SETUP.md`
- **Full Test Suite**: See `TEST_SUITE.md`
- **Performance Analysis**: See `PERFORMANCE_COMPARISON.md`
- **Main Instructions**: See `CLAUDE.md`

---

## ✅ Quick Checklist

```
□ Run: python3 validate_setup.py
□ Review output and API key status
□ Run: python3 test_method_compare.py
□ Pick best method based on results
□ Get API key if using Gemini/Google Vision
□ Run: python3 extract_floor_plan*.py --image floor5.jpg
□ Check output JSON and SVG
□ Ready to process your images!
```

---

## 🎓 Examples

### Local OCR (FREE, slow)
```bash
time python3 extract_floor_plan.py \
  --image floor5.jpg \
  --tessdata-dir ./tessdata \
  --ocr-engine paddle
# ~40 seconds
```

### Gemini Flash (FAST, cheap) ⚡
```bash
export GEMINI_API_KEY="your-key"
time python3 extract_floor_plan_gemini.py \
  --image floor5.jpg
# <1 second
```

### Google Vision (FAST, moderate cost)
```bash
export GOOGLE_APPLICATION_CREDENTIALS="credentials.json"
time python3 extract_floor_plan_google_vision.py \
  --image floor5.jpg
# 1-3 seconds
```

### Parallel Processing (8 workers)
```bash
# Using Gemini (fastest)
python3 -c "
from pathlib import Path
from multiprocessing import Pool
from extract_floor_plan_gemini import extract_floor_plan_fast

images = list(Path('./floor').glob('*.jpg'))
with Pool(8) as p:
    p.starmap(extract_floor_plan_fast,
              [(img, Path(f'results/{img.stem}.json'),
                Path(f'results/{img.stem}.svg}')) for img in images])
"
# Process 9 images in ~10 seconds!
```

---

**TL;DR**: Run `validate_setup.py` → `test_method_compare.py` → pick method → extract!
