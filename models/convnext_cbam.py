import torch
import torch.nn as nn
import timm 

# CBAM module
from models.cbam import CBAM

class ConvNeXt_CBAM(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeXt_CBAM, self).__init__()
        self.model = timm.create_model("convnext_base", pretrained=True)

        self.cbam1 = CBAM(in_channels=128)
        self.cbam2 = CBAM(in_channels=256)
        self.cbam3 = CBAM(in_channels=512)
        self.cbam4 = CBAM(in_channels=1024)

        #BatchNorm for the last CBAM
        self.norm4 = nn.BatchNorm2d(1024)

        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.model.num_features

        # Replace classification head
        self.model.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.model.stem(x)  # Initial Conv + LayerNorm

        # Stage 1 + CBAM + Residual
        x = self.model.stages[0](x)
        x = x + self.cbam1(x)

        # Stage 2 + CBAM + Residual
        x = self.model.stages[1](x)
        x = x+self.cbam2(x)

        # Stage 3 + CBAM + Residual
        x = self.model.stages[2](x)
        x = x+ self.cbam3(x)

        # Final stage + CBAM + Residual + BatchNorm
        x = self.model.stages[3](x)
        x = x+self.cbam4(x)
        x = self.norm4(x)  # BatchNorm2d

        # Pooling and classification
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.model.head(x)

        return x
