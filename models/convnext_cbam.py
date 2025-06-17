import torch
import torch.nn as nn
import timm 

# CBAM module
from models.cbam import CBAM

class ConvNeXt_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_CBAM, self).__init__()
        
        # Load pre-trained ConvNeXt base model
        self.model = timm.create_model("convnext_base", pretrained=True)

        # Define CBAM modules for each stage, matching the output channels of ConvNeXt base
        self.cbam1 = CBAM(in_channels=128)
        self.cbam2 = CBAM(in_channels=256)
        self.cbam3 = CBAM(in_channels=512)
        self.cbam4 = CBAM(in_channels=1024)

        #BatchNorm for the last CBAM
        self.norm4 = nn.BatchNorm2d(1024)

        # Define global average pooling to reduce spatial dimensions to 1x1
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Replace classification head
        in_features = self.model.num_features # 1024 for ConvNeXt base
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        #Stem stage: Initial Conv + LayerNorm
        x = self.model.stem(x)  

        # Stage 1 + CBAM + Residual
        x = self.model.stages[0](x)
        x = x + self.cbam1(x)

        # Stage 2 + CBAM + Residual
        x = self.model.stages[1](x)
        x = x+self.cbam2(x)

        # Stage 3 + CBAM + Residual
        x = self.model.stages[2](x)
        x = x+ self.cbam3(x)

        # Stage 4 + CBAM + Residual + BatchNorm
        x = self.model.stages[3](x)
        x = x+self.cbam4(x)
        x = self.norm4(x)  # BatchNorm2d

        # Apply global average pooling to reduce spatial dimensions
        x = self.global_pool(x)
        
        # Flatten the output for the classification head
        x = torch.flatten(x, 1)
        
        # Apply the classification head to produce logits
        x = self.model.head(x)

        return x
