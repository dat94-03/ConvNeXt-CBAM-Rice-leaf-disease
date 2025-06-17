import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from convnext_cbam import ConvNeXt_CBAM
from torchinfo import summary

# Initialize model
model = ConvNeXt_CBAM(num_classes=6)

# Generate the summary object
model_summary = summary(
    model,
    input_size=(1, 3, 224, 224),
    col_names=["input_size", "output_size", "num_params", "kernel_size", "mult_adds"],
    verbose=0  # concise output
)

# Print the summary
print(model_summary)
