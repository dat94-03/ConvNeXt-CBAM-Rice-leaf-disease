import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from convnext_cbam import ConvNeXt_CBAM
from torchinfo import summary

model = ConvNeXt_CBAM(num_classes=10)

summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"]
)
