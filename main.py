from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import torch
torch.set_default_tensor_type('torch.FloatTensor')  # force CPU-only mode
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
from torchvision.models import resnet50

app = FastAPI(title="Sugarcane Disease Classifier")

# === PASTE YOUR FULL SE_ResNet50 CLASS DEFINITION HERE ===
# Copy everything from class SE_ResNet50(nn.Module): to the end of forward()
class SE_ResNet50(nn.Module):
    """
    Your Squeeze-and-Excitation modified ResNet50 model.
    Adjust the SE blocks / modifications to match what you actually trained.
    """
    def __init__(self, num_classes=5):  # ← change to your actual number of classes
        super(SE_ResNet50, self).__init__()
        
        # Load base ResNet50 (without pretrained weights if you fine-tuned from scratch)
        self.base = resnet50(weights=None)  # or 'IMAGENET1K_V1' if you started pretrained
        
        # === ADD YOUR SE (Squeeze-Excitation) MODIFICATIONS HERE ===
        # Example placeholder — replace with your actual SE implementation
        # self.base.layer1 = YourSEBlock(self.base.layer1)
        # self.base.layer2 = YourSEBlock(self.base.layer2)
        # etc.
        
        # Final classifier head (must match your training)
        in_features = self.base.fc.in_features
        self.base.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.base(x)

# Load model once at startup
device = torch.device('cpu')  # Vercel runs on CPU
model = SE_ResNet50(num_classes=5)

# <--- THIS IS THE LINE YOU'RE LOOKING FOR --->
#weights_path = "model_weights.pth"  # or your original filename
#model.load_state_dict(torch.load(weights_path, map_location=device))
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

#IMPORT THE MODEL FROM GDRIVE
import os
import gdown

# Get URL from Render env var (you set this on dashboard)
weights_url = os.getenv("WEIGHTS_URL")

if not weights_url:
    raise RuntimeError("WEIGHTS_URL environment variable is not set on Render!")

print(f"Downloading model weights from: {weights_url}")

try:
    gdown.download(weights_url, "model_weights.pth", quiet=False)
except Exception as e:
    raise RuntimeError(f"Failed to download weights: {str(e)}")

try:
    state_dict = torch.load("model_weights.pth", map_location=device)
    model.load_state_dict(state_dict)
except Exception as e:
    raise RuntimeError(f"Failed to load state_dict: {str(e)}")

model.to(device)
model.eval()

print("Model loaded successfully!")

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class_names = ['healthy', 'mosaic', 'redrot', 'rust', 'smut']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred_idx = probs.max(dim=0)

        class_name = class_names[pred_idx.item()]
        conf_pct = conf.item() * 100

        advice = {
            'healthy': "Leaf looks healthy! Keep monitoring.",
            'mosaic': "Mosaic virus detected. Remove affected leaves, control aphids.",
            'redrot': "Red rot detected. Apply fungicide and improve drainage.",
            'rust': "Rust detected. Use rust-specific fungicide.",
            'smut': "Smut detected. Remove and destroy infected material."
        }.get(class_name, "Unknown — consult expert.")

        result = {
            "disease": class_name.capitalize(),
            "confidence": f"{conf_pct:.1f}%",
            "advice": advice
        }

        if conf_pct < 80:
            result["warning"] = "Low confidence — photo may be unclear. Try again."

        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
