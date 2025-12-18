# SOTA Image Classification Models - Summary

**Date:** 2025-12-18  
**Server:** torch-inference v1.0.0  
**Total Models:** 12 Image Classification Models

## Quick Reference

### By Size (Smallest to Largest)

| # | Model | Size | Accuracy | Best For |
|---|-------|------|----------|----------|
| 1 | MobileNetV4 Hybrid Large | 140 MB | 84.36% | ⚡ Mobile/Edge |
| 2 | CoAtNet-3 | 700 MB | 86.0% | Efficient hybrid |
| 3 | Swin Transformer Large | 790 MB | 87.3% | Hierarchical vision |
| 4 | EfficientNetV2 XL | 850 MB | 87.3% | Speed/accuracy balance |
| 5 | EVA-02 Large | 1.2 GB | 90.054% | 🥇 **Highest accuracy** |
| 6 | BEiT Large | 1.2 GB | 88.6% | BERT-style vision |
| 7 | MaxViT XLarge | 1.8 GB | 88.53% | Hybrid architecture |
| 8 | DeiT-III Huge | 2.5 GB | 87.7% | Data-efficient |
| 9 | ConvNeXt V2 Huge | 2.6 GB | 88.848% | Best ConvNet |
| 10 | ConvNeXt XXLarge CLIP | 3.5 GB | 88.612% | Zero-shot/CLIP |
| 11 | EVA Giant | 4.0 GB | 89.792% | Research |
| 12 | ViT Giant CLIP | 5.0 GB | 88.5% | Largest ViT |

### By Accuracy (Highest to Lowest)

| # | Model | Accuracy | Size | Architecture |
|---|-------|----------|------|--------------|
| 1 | **EVA-02 Large** | 90.054% | 1.2 GB | Vision Transformer |
| 2 | EVA Giant | 89.792% | 4.0 GB | Vision Transformer |
| 3 | ConvNeXt V2 Huge | 88.848% | 2.6 GB | ConvNet |
| 4 | ConvNeXt XXLarge CLIP | 88.612% | 3.5 GB | ConvNet |
| 5 | BEiT Large | 88.6% | 1.2 GB | BEiT |
| 6 | MaxViT XLarge | 88.53% | 1.8 GB | Hybrid |
| 7 | ViT Giant CLIP | 88.5% | 5.0 GB | Vision Transformer |
| 8 | DeiT-III Huge | 87.7% | 2.5 GB | DeiT |
| 9 | EfficientNetV2 XL | 87.3% | 850 MB | EfficientNet |
| 10 | Swin Transformer Large | 87.3% | 790 MB | Swin |
| 11 | CoAtNet-3 | 86.0% | 700 MB | Hybrid |
| 12 | MobileNetV4 Hybrid Large | 84.36% | 140 MB | MobileNet |

## API Endpoints

### List All SOTA Models
```bash
curl http://localhost:8000/models/sota
```

### Download Specific Model
```bash
# Smallest model (recommended for testing)
curl -X POST http://localhost:8000/models/sota/mobilenetv4-hybrid-large

# Highest accuracy
curl -X POST http://localhost:8000/models/sota/eva02-large-patch14-448

# Best balance
curl -X POST http://localhost:8000/models/sota/efficientnetv2-xl
```

### Check Download Status
```bash
curl http://localhost:8000/models/download/list
```

## Recommendations

### 🎯 For Production (Best Overall)
**EVA-02 Large**
- Highest accuracy: 90.054%
- Reasonable size: 1.2 GB
- Excellent performance

### ⚡ For Mobile/Edge
**MobileNetV4 Hybrid Large**
- Smallest: 140 MB
- Fast inference
- Still 84.36% accuracy

### 💪 For Research
**EVA Giant** or **ViT Giant CLIP**
- State-of-the-art results
- Largest models available
- Comprehensive features

### ⚖️ Best Balance
**EfficientNetV2 XL** or **Swin Transformer Large**
- Good accuracy: 87.3%
- Moderate size: 700-850 MB
- Fast inference

## Architecture Types

### Vision Transformers (ViT)
- EVA-02 Large (90.054%)
- EVA Giant (89.792%)
- BEiT Large (88.6%)
- ViT Giant CLIP (88.5%)
- DeiT-III Huge (87.7%)

### Pure ConvNets
- ConvNeXt V2 Huge (88.848%)
- ConvNeXt XXLarge CLIP (88.612%)
- EfficientNetV2 XL (87.3%)
- MobileNetV4 (84.36%)

### Hybrid (Conv + Attention)
- MaxViT XLarge (88.53%)
- CoAtNet-3 (86.0%)
- Swin Transformer Large (87.3%)

## Current Status

✅ **Working:**
- Model registry and listing
- Download API endpoints
- Download progress tracking
- Model metadata retrieval

⚠️ **Requires `--features torch` rebuild:**
- Actual inference
- GPU acceleration
- Model loading

## Test Commands

```bash
# 1. Check server health
curl http://localhost:8000/health

# 2. List all available models
curl http://localhost:8000/models/available | jq '.models[] | select(.task == "Image Classification")'

# 3. Get SOTA models only
curl http://localhost:8000/models/sota | jq '.models[] | {id, name, accuracy, size}'

# 4. Download smallest model
curl -X POST http://localhost:8000/models/sota/mobilenetv4-hybrid-large | jq .

# 5. Check download status
curl http://localhost:8000/models/download/list | jq .
```

## All Model IDs

For quick reference when calling the API:

```
eva02-large-patch14-448
eva-giant-patch14-560
convnextv2-huge-512
convnext-xxlarge-clip
maxvit-xlarge-512
vit-giant-patch14-224
beit-large-patch16-512
deit3-huge-patch14-224
efficientnetv2-xl
swin-large-patch4-384
coatnet-3-rw-224
mobilenetv4-hybrid-large
```

---

**For detailed API documentation, see:** [API_SOTA_MODELS.md](API_SOTA_MODELS.md)  
**For testing results, see:** [TEST_RESULTS.md](TEST_RESULTS.md)
