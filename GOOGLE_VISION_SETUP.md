# Google Cloud Vision Setup - Fast Floor Plan Extraction

## Why Google Cloud Vision?

| Metric | Local OCR | Google Vision |
|--------|-----------|---------------|
| Speed | 30-40s/image | **1-3s/image** |
| Accuracy | Good | **Excellent** |
| Cost | Free | ~$0.0006/image |
| Setup | Easy | Moderate |

**Result: 10-40x faster processing!**

## Setup Instructions

### 1. Create Google Cloud Project

```bash
# Install gcloud CLI
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init

# Create project
gcloud projects create floor-plan-extractor
gcloud config set project floor-plan-extractor
```

### 2. Enable Vision API

```bash
# Enable the Vision API
gcloud services enable vision.googleapis.com

# Create service account
gcloud iam service-accounts create floor-plan-sa \
  --display-name="Floor Plan Extractor"

# Grant Vision API permissions
gcloud projects add-iam-policy-binding floor-plan-extractor \
  --member=serviceAccount:floor-plan-sa@floor-plan-extractor.iam.gserviceaccount.com \
  --role=roles/ml.admin
```

### 3. Create API Key

```bash
# Create and download JSON key
gcloud iam service-accounts keys create credentials.json \
  --iam-account=floor-plan-sa@floor-plan-extractor.iam.gserviceaccount.com

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="$(pwd)/credentials.json"
```

### 4. Install Dependencies

```bash
pip install google-cloud-vision
```

### 5. Run Fast Extraction

```bash
# Single image (1-3 seconds)
python3 extract_floor_plan_google_vision.py \
  --image ./floor5.jpg \
  --json-out floorplan_state.json \
  --svg-out final.svg

# Batch processing with parallel
find ./floor -name "*.jpg" -o -name "*.png" | \
  parallel -j 8 python3 extract_floor_plan_google_vision.py --image {} --json-out {.}.json
```

## Expected Performance

**Before (Local OCR):**
```
9 images × 33.5s = 301.5 seconds total
```

**After (Google Vision + Parallel):**
```
9 images × 2s × 8 workers = ~2.25 seconds total
100x faster! 🚀
```

## Cost Estimate

- **100 images/month**: ~$0.06
- **1,000 images/month**: ~$0.60
- **10,000 images/month**: ~$6.00

Free tier: 1,000 requests/month

## Troubleshooting

```bash
# Test credentials
python3 -c "from google.cloud import vision; print('✓ Google Vision installed')"

# Verify API enabled
gcloud services list --enabled | grep vision

# Check quota
gcloud compute project-info describe --project=floor-plan-extractor
```

## Fallback to Local OCR

If Google Vision fails, the script automatically falls back to local OCR.

## Alternative: AWS Textract

If you prefer AWS:

```bash
pip install boto3

# Similar speed (2-5s/image)
# Cost: $1.50-4 per 1,000 images
```
