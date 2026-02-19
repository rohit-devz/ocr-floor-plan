# Floor Plan Extraction - Optimization Guide

## Current Performance
- Average: **15.37s per image**
- Range: 3.5s - 43s depending on image size/complexity
- Main bottleneck: **OCR initialization and multiple preprocessing passes**

## Optimization Strategies

### 1. **Skip Dimension Extraction (Fastest - ~40% reduction)**
If you don't need room dimensions, disable them:

**Code change in `extract_floor_plan.py` (Optional flags needed):**
```python
# Add flag: --skip-dimensions
# Then in extract_floorplan():
if not args.skip_dimensions:
    # Run dimension extraction
```

**Estimated speedup:** 40-50% (saves dimension parsing)

### 2. **Reduce Image Size (30-40% reduction)**
Process smaller images when resolution isn't critical:

```bash
# Resize image before processing
convert floor5.jpg -resize 50% floor5_small.jpg
python3 extract_floor_plan.py --image floor5_small.jpg
```

**Estimated speedup:** 30-40%

### 3. **Simplify OCR Preprocessing (20-30% reduction)**
The current code tries 8 different preprocessing approaches. You could reduce this.

**In `_ocr_tokens_tesseract()` (line 212-221):**
```python
# Current: 8 passes
passes = [
    (image, 1.0, cfg_base + " --psm 11"),
    (image, 1.0, cfg_base + " --psm 6"),
    # ... 6 more ...
]

# Optimized: 2-3 passes
passes = [
    (image, 1.0, cfg_base + " --psm 6"),  # Usually works best
    (gray2x, 2.0, cfg_base + " --psm 11"),  # Fallback
]
```

**Estimated speedup:** 20-30%

### 4. **Batch Processing with Parallel Execution**
Process multiple images in parallel (if you have multiple cores):

```bash
# Using GNU parallel
find ./floor -name "*.jpg" | parallel python3 extract_floor_plan.py --image {}

# Or using Python multiprocessing (update test script)
```

**Estimated speedup:** 4-8x (with 4-8 cores)

### 5. **Cache OCR Results (50-70% reduction on subsequent runs)**
If processing the same image multiple times:

```python
import hashlib
from pathlib import Path

def get_ocr_cache_path(image_path):
    h = hashlib.md5(open(image_path, 'rb').read()).hexdigest()
    return Path(f"./.ocr_cache/{h}.json")

# In ocr_tokens():
cache_path = get_ocr_cache_path(image)
if cache_path.exists():
    return json.load(open(cache_path))
# ... run OCR ...
json.dump(tokens, open(cache_path, 'w'))
```

**Estimated speedup:** 50-70% (on repeated runs)

## Quick Wins (Implement First)

1. **Skip dimensions if not needed** → 40-50% faster
2. **Reduce image size** → 30-40% faster
3. **Simplify OCR passes** → 20-30% faster
4. **Use parallel processing** → 4-8x faster (with multiple cores)

## Recommended Configuration for Speed

```bash
# Fast mode - skip dimensions, reduce preprocessing
python3 extract_floor_plan.py \
  --image floor5.jpg \
  --ocr-engine paddle \
  --skip-dimensions
```

**Expected time: 6-8s per image** (vs 15s currently)

## Profiling

To find the exact bottleneck in YOUR setup:

```bash
python3 -m cProfile -s cumulative extract_floor_plan.py --image floor5.jpg 2>&1 | head -30
```

This will show which functions take the most time.
