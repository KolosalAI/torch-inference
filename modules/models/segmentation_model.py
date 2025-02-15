import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
from modules.utils.config import SEGMENTATION_CONFIG

class DoubleConv(nn.Module):
    """(Conv2d => BN => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        mid_channels = mid_channels or out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))
        
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, 
                 n_classes: int = 1,
                 input_channels: int = 3,
                 bilinear: bool = True,
                 trt_optimized: bool = False):
        super(UNet, self).__init__()
        self.n_classes = n_classes
        self.input_channels = input_channels
        self.bilinear = bilinear
        self.trt_optimized = trt_optimized

        self.inc = DoubleConv(input_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if trt_optimized:
            self._optimize_for_tensorrt()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        if self.trt_optimized:
            return torch.sigmoid(logits) if self.n_classes == 1 else F.softmax(logits, dim=1)
        return logits

    def _optimize_for_tensorrt(self):
        """Optimize model architecture for TensorRT conversion"""
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
                module.track_running_stats = True
                
        self.eval()
        self.scripted = torch.jit.script(self)
        
    def trace_for_trt(self, example_input):
        """Create traced model for TensorRT"""
        return torch.jit.trace(self.scripted, example_input)

class SegmentationCNN(nn.Module):
    """Simpler segmentation model for faster inference"""
    def __init__(self, 
                 input_channels: int = 3,
                 num_classes: int = 1,
                 base_channels: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*2, base_channels, 2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels, num_classes, 2, stride=2),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return self.decoder(x)

def create_segmentation_model(
    model_type: str = "unet",
    pretrained: bool = False,
    input_size: Tuple[int, int] = SEGMENTATION_CONFIG["input_size"],
    num_classes: int = 1,
    trt_optimized: bool = SEGMENTATION_CONFIG["use_tensorrt"]
) -> nn.Module:
    """
    Factory function for creating segmentation models
    
    Args:
        model_type: 'unet' or 'cnn'
        pretrained: Load pretrained weights
        input_size: Expected input size (H, W)
        num_classes: Number of output classes
        trt_optimized: Optimize model for TensorRT conversion
        
    Returns:
        Initialized segmentation model
    """
    input_channels = 3  # RGB
    
    if model_type.lower() == "unet":
        model = UNet(
            n_classes=num_classes,
            input_channels=input_channels,
            trt_optimized=trt_optimized
        )
    elif model_type.lower() == "cnn":
        model = SegmentationCNN(
            input_channels=input_channels,
            num_classes=num_classes
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    if pretrained:
        model = _load_pretrained_weights(model, model_type, input_size)
        
    # Convert to TensorRT compatible format if needed
    if trt_optimized:
        example_input = torch.randn(1, input_channels, *input_size)
        model = model.trace_for_trt(example_input) if isinstance(model, UNet) else torch.jit.script(model)
        
    return model

def _load_pretrained_weights(model: nn.Module, 
                            model_type: str,
                            input_size: Tuple[int, int]) -> nn.Module:
    """Load pretrained weights (stub for actual implementation)"""
    # In practice, you would load weights from a checkpoint here
    print(f"Loading pretrained weights for {model_type}...")
    return model

# Example usage
if __name__ == "__main__":
    model = create_segmentation_model(
        model_type="unet",
        num_classes=1,
        trt_optimized=SEGMENTATION_CONFIG["use_tensorrt"]
    )
    
    example_input = torch.randn(1, 3, *SEGMENTATION_CONFIG["input_size"])
    output = model(example_input)
    print(f"Model output shape: {output.shape}")
    print(f"Output range: [{output.min():.2f}, {output.max():.2f}]")