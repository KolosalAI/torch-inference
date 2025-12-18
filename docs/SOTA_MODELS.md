# SOTA Image Classification Models

## Overview
Added 12 state-of-the-art (SOTA) image classification models covering different architectures and use cases, from highest accuracy to edge-optimized models.

## Models Added

### Tier 1: Highest Accuracy (>89% Top-1)

#### 1. EVA-02 Large (Rank #1)
- **Accuracy**: 90.054% Top-1 ImageNet-1K
- **Architecture**: Vision Transformer (ViT)
- **Size**: ~1.2 GB
- **URL**: https://huggingface.co/timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k
- **Use Case**: Maximum accuracy, research applications
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

#### 2. EVA Giant (Rank #2)
- **Accuracy**: 89.792% Top-1 ImageNet-1K
- **Architecture**: Vision Transformer (ViT) - Very Large
- **Size**: ~4.0 GB
- **URL**: https://huggingface.co/timm/eva_giant_patch14_560.m30m_ft_in22k_in1k
- **Use Case**: Ultimate accuracy when resources allow
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

### Tier 2: High Accuracy ConvNets (88-89% Top-1)

#### 3. ConvNeXt V2 Huge (Rank #3)
- **Accuracy**: 88.848% Top-1 ImageNet-1K
- **Architecture**: Pure ConvNet (modernized)
- **Size**: ~2.6 GB
- **URL**: https://huggingface.co/timm/convnextv2_huge.fcmae_ft_in22k_in1k_512
- **Use Case**: Best ConvNet architecture, excellent for dense prediction
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

#### 4. ConvNeXt XXLarge CLIP (Rank #4)
- **Accuracy**: 88.612% Top-1 ImageNet-1K
- **Architecture**: Pure ConvNet
- **Size**: ~3.5 GB
- **URL**: https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k
- **Use Case**: CLIP-pretrained, excellent for transfer learning
- **Pretraining**: CLIP on LAION-2B → ImageNet-1K fine-tuning

### Tier 3: Hybrid Models (88-89% Top-1)

#### 5. MaxViT XLarge (Rank #5)
- **Accuracy**: 88.53% Top-1 ImageNet-1K
- **Architecture**: Hybrid (Conv + Multi-axis Attention)
- **Size**: ~1.8 GB
- **URL**: https://huggingface.co/timm/maxvit_xlarge_tf_512.in21k_ft_in1k
- **Use Case**: Best hybrid architecture, good efficiency/accuracy tradeoff
- **Pretraining**: ImageNet-21K → ImageNet-1K fine-tuning

#### 10. CoAtNet-3 (Rank #10)
- **Accuracy**: ~86.0% Top-1 ImageNet-1K
- **Architecture**: Hybrid (Conv + Attention)
- **Size**: ~700 MB
- **URL**: https://huggingface.co/timm/coatnet_3_rw_224.sw_in1k
- **Use Case**: Efficient hybrid, good for various resolutions
- **Pretraining**: ImageNet-1K only

### Tier 4: ViT Variants (87-88.6% Top-1)

#### 6. ViT Giant CLIP (Rank #6)
- **Accuracy**: ~88.5% Top-1
- **Architecture**: Vision Transformer
- **Size**: ~5.0 GB
- **URL**: https://huggingface.co/timm/vit_giant_patch14_224.clip_laion2b
- **Use Case**: CLIP pretraining, excellent zero-shot capabilities
- **Pretraining**: CLIP on LAION-2B

#### 7. BEiT Large (Rank #7)
- **Accuracy**: 88.6% Top-1 ImageNet-1K
- **Architecture**: Vision Transformer (BERT-style pretraining)
- **Size**: ~1.2 GB
- **URL**: https://huggingface.co/timm/beit_large_patch16_512.in22k_ft_in22k_in1k
- **Use Case**: Self-supervised pretraining approach
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

#### 8. DeiT-III Huge (Rank #8)
- **Accuracy**: ~87.7% Top-1 ImageNet-1K
- **Architecture**: Vision Transformer (Data-efficient)
- **Size**: ~2.5 GB
- **URL**: https://huggingface.co/timm/deit3_huge_patch14_224.fb_in22k_ft_in1k
- **Use Case**: Efficient training, good generalization
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

#### 9. Swin Transformer Large (Rank #9)
- **Accuracy**: 87.3% Top-1 ImageNet-1K
- **Architecture**: Hierarchical Vision Transformer
- **Size**: ~790 MB
- **URL**: https://huggingface.co/timm/swin_large_patch4_window12_384.ms_in22k_ft_in1k
- **Use Case**: Excellent for dense prediction tasks (detection, segmentation)
- **Pretraining**: ImageNet-22K → ImageNet-1K fine-tuning

### Tier 5: Efficient Models

#### 8. EfficientNetV2 XL (Rank #8)
- **Accuracy**: 87.3% Top-1 ImageNet-1K
- **Architecture**: EfficientNet (compound scaling)
- **Size**: ~850 MB
- **URL**: https://huggingface.co/timm/tf_efficientnetv2_xl.in21k_ft_in1k
- **Use Case**: Excellent accuracy/speed tradeoff
- **Pretraining**: ImageNet-21K → ImageNet-1K fine-tuning

#### 13. MobileNetV4 Hybrid Large (Rank #13)
- **Accuracy**: 84.36% Top-1 ImageNet-1K
- **Architecture**: MobileNet (edge-optimized)
- **Size**: ~140 MB
- **URL**: https://huggingface.co/timm/mobilenetv4_hybrid_large.e600_r448_in1k
- **Use Case**: Mobile/edge deployment, real-time inference
- **Pretraining**: ImageNet-1K only

## Usage Recommendations

### For Maximum Accuracy
1. **EVA-02 Large** (90.05%) - Best overall
2. **EVA Giant** (89.79%) - Largest model
3. **ConvNeXt V2 Huge** (88.85%) - Best ConvNet

### For Transfer Learning
1. **ConvNeXt XXLarge CLIP** - CLIP pretraining
2. **ViT Giant CLIP** - Zero-shot capabilities
3. **BEiT Large** - Self-supervised features

### For Dense Prediction (Detection/Segmentation)
1. **Swin Transformer Large** - Hierarchical features
2. **ConvNeXt V2 Huge** - Strong spatial features
3. **MaxViT XLarge** - Multi-scale features

### For Production Deployment
1. **EfficientNetV2 XL** - Best efficiency/accuracy
2. **MaxViT XLarge** - Hybrid efficiency
3. **CoAtNet-3** - Smaller, still accurate

### For Mobile/Edge
1. **MobileNetV4 Hybrid Large** - Optimized for edge
2. **EfficientNetV2 XL** - Can be quantized

## Download Instructions

All models can be loaded using the `timm` library:

```python
import timm

# Load model
model = timm.create_model('eva02_large_patch14_448.mim_m38m_ft_in22k_in1k', pretrained=True)

# Or download from Hugging Face
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="timm/eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
    filename="pytorch_model.bin"
)
```

## Model Registry Integration

All models have been added to `model_registry.json` with the following fields:
- `status`: "Available"
- `architecture`: Model architecture type
- `task`: "Image Classification"
- `accuracy`: ImageNet-1K Top-1 accuracy
- `rank`: Performance ranking
- `size`: Approximate model size
- `url`: Hugging Face model URL
- `dataset`: Training/pretraining datasets
- `name`: Display name

## Performance Comparison

| Rank | Model | Accuracy | Size | Architecture | Best For |
|------|-------|----------|------|--------------|----------|
| 1 | EVA-02 Large | 90.05% | 1.2 GB | ViT | Maximum accuracy |
| 2 | EVA Giant | 89.79% | 4.0 GB | ViT | Research |
| 3 | ConvNeXt V2 Huge | 88.85% | 2.6 GB | ConvNet | Dense prediction |
| 4 | ConvNeXt XXL CLIP | 88.61% | 3.5 GB | ConvNet | Transfer learning |
| 5 | MaxViT XLarge | 88.53% | 1.8 GB | Hybrid | Balanced |
| 6 | ViT Giant CLIP | 88.5% | 5.0 GB | ViT | Zero-shot |
| 7 | BEiT Large | 88.6% | 1.2 GB | ViT | Self-supervised |
| 8 | DeiT-III Huge | 87.7% | 2.5 GB | ViT | Efficient training |
| 8 | EfficientNetV2 XL | 87.3% | 850 MB | EfficientNet | Production |
| 9 | Swin Large | 87.3% | 790 MB | Swin | Detection/Seg |
| 10 | CoAtNet-3 | 86.0% | 700 MB | Hybrid | Efficient |
| 13 | MobileNetV4 | 84.36% | 140 MB | MobileNet | Mobile/Edge |

## Notes

- All accuracies are ImageNet-1K validation set Top-1 accuracy
- Models with ImageNet-21K/22K pretraining generally perform better
- CLIP-pretrained models have excellent transfer learning capabilities
- Size estimates include model weights only (not including runtime overhead)
- All models available on Hugging Face Hub with Apache 2.0 or similar licenses

## References

- timm library: https://github.com/huggingface/pytorch-image-models
- Hugging Face Model Hub: https://huggingface.co/timm
- Papers with Code ImageNet Leaderboard: https://paperswithcode.com/sota/image-classification-on-imagenet
