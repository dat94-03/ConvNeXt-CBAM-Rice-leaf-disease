import torch
import torch.nn as nn
import timm  # Import timm for ConvNeXt

# CBAM module
from models.cbam import CBAM

class ConvNeXt_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_CBAM, self).__init__()
        self.model = timm.create_model("convnext_base", pretrained=True)

        # CBAM modules at different feature extraction stages
        self.cbam1 = CBAM(in_channels=128)
        self.cbam2 = CBAM(in_channels=256)
        self.cbam3 = CBAM(in_channels=512)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.model.num_features

        # Replace classification head
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.stem(x)  # Initial Conv + LayerNorm

        x = self.model.stages[0](x)  # First feature stage
        x = self.cbam1(x)            # Apply CBAM 

        x = self.model.stages[1](x)  # Second feature stage
        x = self.cbam2(x)

        x = self.model.stages[2](x)  # Third feature stage
        x = self.cbam3(x)

        x = self.model.stages[3](x)  # Final feature stage (1024 channels)

        x = self.global_pool(x)  # Ensure pooling before classifier
        x = torch.flatten(x, 1)  # Flatten for FC layer
        x = self.model.head(x)  # Final classification

        return x
