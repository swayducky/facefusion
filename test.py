import torch
import torch.nn as nn
import onnx
import onnxruntime

# Create a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 10)

    def forward(self, x):
        return self.fc(x)

model = SimpleModel()

# Convert the PyTorch model to ONNX
dummy_input = torch.randn(1, 10)
torch.onnx.export(model, dummy_input, "simple_model.onnx")

# Test the ONNX model using onnxruntime with GPU
sess = onnxruntime.InferenceSession("simple_model.onnx", providers=['CUDAExecutionProvider'])

# Check if the runtime is using GPU
print("Providers: ", sess.get_providers())

# Inference
input_name = sess.get_inputs()[0].name
pred_onnx = sess.run(None, {input_name: dummy_input.numpy()})

print("Output: ", pred_onnx)
