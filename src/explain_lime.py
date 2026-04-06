import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

def run_lime(model, image_path):

    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])

    image = Image.open(image_path).convert("RGB")
    image_np = np.array(image.resize((224,224)))

    def batch_predict(images):
        batch = torch.stack([
            transform(Image.fromarray(img)) for img in images
        ])

        with torch.no_grad():
            logits = model(batch)

        return torch.nn.functional.softmax(logits, dim=1).numpy()

    explainer = lime_image.LimeImageExplainer()

    explanation = explainer.explain_instance(
        image_np,
        batch_predict,
        top_labels=5,
        hide_color=0,
        num_samples=1000
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=5
    )

    plt.imshow(mark_boundaries(temp/255.0, mask))
    plt.axis("off")
    
    import os
    os.makedirs("outputs", exist_ok=True)
    plt.savefig("outputs/lime_result.png")
    plt.close()
