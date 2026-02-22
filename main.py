from fastapi import FastAPI, UploadFile, File
from PIL import Image
import tensorflow as tf
import torch
from torchvision import models, transforms
import numpy as np
import json
import shutil
import re
import os
import gdown

app = FastAPI(title="Crop Disease AI API")

# =====================================================
# DOWNLOAD MODELS FROM GOOGLE DRIVE
# =====================================================

FILES = {
    "plantvillage_resnet50.keras": "1pNWvufq5rCLn0Amov1zAQBYCHRmlk85b",
    "bd_model.pth": "1OAL4LX7BmR8G_oP3LJWfgsxmT_biXRAK",
    "class_mapping.json": "1YSShfJqlcSWFKX9Nvon7LH0nPStaYO8v",
    "labels.json": "1UgHKFAzym9Qwgk3go5y70EnNRWOCQ3vD"
}

def download_models():
    for file, file_id in FILES.items():
        if not os.path.exists(file):
            print(f"Downloading {file}...")
            gdown.download(
                f"https://drive.google.com/uc?id={file_id}",
                file,
                quiet=False
            )

download_models()

# =====================================================
# LOAD PLANTVILLAGE MODEL
# =====================================================

pv_model = tf.keras.models.load_model("plantvillage_resnet50.keras")

with open("labels.json") as f:
    pv_classes = json.load(f)

def predict_pv(image_path):
    img = Image.open(image_path).convert("RGB").resize((224,224))
    img = np.array(img)/255.0
    img = np.expand_dims(img,0)

    preds = pv_model.predict(img, verbose=0)[0]
    conf = float(np.max(preds))
    idx = int(np.argmax(preds))

    label = re.sub("_+"," ", pv_classes[idx]).title()
    return label, conf


# =====================================================
# LOAD BD MODEL
# =====================================================

model_bd = models.resnet50(weights=None)

num_ftrs = model_bd.fc.in_features
model_bd.fc = torch.nn.Sequential(
    torch.nn.Linear(num_ftrs,512),
    torch.nn.ReLU(),
    torch.nn.Dropout(0.2),
    torch.nn.Linear(512,94)
)

checkpoint = torch.load(
    "bd_model.pth",
    map_location="cpu"
)

state_dict = checkpoint.get("model_state_dict", checkpoint)
state_dict = {k.replace("module.",""):v for k,v in state_dict.items()}

model_bd.load_state_dict(state_dict)
model_bd.eval()

with open("class_mapping.json") as f:
    bd_classes = json.load(f)

transform_bd = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

def predict_bd(image_path):
    img = Image.open(image_path).convert("RGB")
    tensor = transform_bd(img).unsqueeze(0)

    with torch.no_grad():
        outputs = model_bd(tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)[0]

    conf, idx = torch.max(probs,0)
    label = re.sub("_+"," ", bd_classes[str(idx.item())]).title()

    return label, float(conf.item())


# =====================================================
# DUAL MODEL PREDICTION
# =====================================================

def dual_predict(image_path):

    pv_label, pv_conf = predict_pv(image_path)
    bd_label, bd_conf = predict_bd(image_path)

    if pv_conf > bd_conf:
        return pv_label, pv_conf, "PlantVillage Model"

    return bd_label, bd_conf, "BD Dataset Model"


# =====================================================
# RECOMMENDATION ENGINE
# =====================================================

def recommendation_engine(name):

    name = name.lower()

    if "healthy" in name:
        return ["No disease detected"], ["Apply balanced NPK fertilizer"]

    if "bacterial" in name:
        return ["Spray copper bactericide"], ["Add Zinc + Balanced NPK"]

    if any(k in name for k in ["virus","mosaic","curl"]):
        return ["Remove infected plants"], ["Apply potassium fertilizer"]

    return ["Apply fungicide"], ["Use organic compost"]


# =====================================================
# API ENDPOINT
# =====================================================

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    path = f"temp_{file.filename}"

    with open(path,"wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

    disease, conf, model_used = dual_predict(path)

    treat, fert = recommendation_engine(disease)

    return {
        "status": "success",
        "prediction": {
            "disease": disease,
            "confidence": round(conf*100,2),
            "model_used": model_used
        },
        "recommendations": {
            "treatment": treat,
            "fertilizer": fert
        }

    }
# =====================================================
# RUN SERVER (RENDER PORT FIX)
# =====================================================

if __name__ == "__main__":
    import uvicorn
    import os

    port = int(os.environ.get("PORT", 10000))

    uvicorn.run("main:app", host="0.0.0.0", port=port)
