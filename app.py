import gradio as gr
import torch
import cv2
from torchvision import transforms
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionalNetwork(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3 , 1)

        # Fully connected Layer
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2) # 2x2 kernel and stride 2

        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2) # 2x2 kernel and stride 2

        # Review to flatten it out
        X = X.view(-1, 16*5*5)

        # Fully connected layer
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

model = ConvolutionalNetwork()
model.load_state_dict(torch.load('guide_mnist_model.pt'))
print(model.eval())

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# Function: accepts PIL image, converts to tensor
def predict_mnist(image):
    image_tensor = transform(image)             # Shape: [1, 28, 28]
    image_tensor = image_tensor.unsqueeze(0)    # Shape: [1, 1, 28, 28]
    
    with torch.no_grad():
        prediction = model(image_tensor)

    return str(prediction.argmax().item())

# Gradio interface
demo = gr.Interface(
    fn=predict_mnist,
    inputs=gr.Image(type="pil", label="Upload Image"),  # Receives PIL Image
    outputs=gr.Textbox(label="Predicted Digit")
)

demo.launch()

