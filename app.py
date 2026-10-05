import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import urllib.request
import os

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Saree Pattern Classifier",
    page_icon="🥻",
    layout="centered"
)

st.title("🥻 Indian Saree Pattern Classifier")
st.write("Upload a saree image to identify its pattern.")

# -----------------------------
# Device
# -----------------------------
device = torch.device("cpu")

# -----------------------------
# Model Architecture
# -----------------------------
class SareeNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()

        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()

        self.backbone = backbone

        self.embedding = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )

        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.embedding(features)
        output = self.classifier(embedding)
        return output


# -----------------------------
# Download Model
# -----------------------------
MODEL_PATH = "best_sareenet.pth"

MODEL_URL = (
    "https://github.com/Tharun-06/"
    "saree-pattern-classifier/releases/download/"
    "v1.0/best_sareenet.pth"
)

@st.cache_resource
def load_model():

    if not os.path.exists(MODEL_PATH):

        with st.spinner("Downloading trained model..."):
            urllib.request.urlretrieve(
                MODEL_URL,
                MODEL_PATH
            )

    model = SareeNet(num_classes=4)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


model = load_model()

# -----------------------------
# Image Transform
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -----------------------------
# Classes
# -----------------------------
classes = [
    "Banarasi",
    "Bandhani",
    "Ikat",
    "Pichwai"
]

# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a saree image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Saree Image",
        use_container_width=True
    )

    if st.button("🔍 Predict Pattern"):

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)
        image_tensor = image_tensor.to(device)

        with torch.no_grad():

            output = model(image_tensor)

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted_class = torch.max(
                probabilities,
                dim=1
            )

        predicted_name = classes[predicted_class.item()]
        confidence_percentage = confidence.item() * 100

        st.success(
            f"Predicted Saree Pattern: {predicted_name}"
        )

        st.info(
            f"Confidence: {confidence_percentage:.2f}%"
        )
