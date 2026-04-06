import torch
from PIL import Image
import torchvision.transforms as transforms
import json

def predict(model, image_path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output, dim=1)

    confidence, pred = torch.max(probs, 1)

    result = {
        "predicted_class": int(pred.item()),
        "confidence": float(confidence.item())
    }

    with open("outputs/prediction.json", "w") as f:
        json.dump(result, f, indent=4)

    print("Prediction saved.")