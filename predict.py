import os
import time
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
import skimage.io as io
import matplotlib.pyplot as plt

from src.models_vit_tensor_CD_2 import vit_base_patch8


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def open_image(img_path):
    img = io.imread(img_path).astype(np.float32)
    return img


def preprocess_image(img):
    b = np.mean(img, axis=2, keepdims=True)
    img = np.concatenate((img, b, b), axis=2)

    kid = img - img.min(axis=(0, 1), keepdims=True)
    mom = img.max(axis=(0, 1), keepdims=True) - img.min(axis=(0, 1), keepdims=True) + 1e-10
    img = kid / mom

    transform = transforms.ToTensor()
    return transform(img).unsqueeze(0)  # Add batch dim

def map_prediction_to_original(pred):
    """
    Map model prediction (1–12 classes) back to the original mask codes
    and compute class-wise percentages.
    """
    # Reverse of your forward mapping
    reverse_map = {
        0: 0,    # Background
        1: 21,   # Arable land
        2: 22,   # Permanent crops
        3: 23,   # Pastures
        4: 31,   # Forests
        5: 5,    # Surface water
        6: 32,   # Shrub
        7: 33,   # Open spaces
        8: 41,   # Wetlands
        9: 13,   # Mine dump
        10: 14,  # Artificial vegetation
        11: 11,  # Urban fabric
        12: 12   # Buildings
    }

    # Category descriptions
    category_desc = {
        0: "Background",
        21: "Arable land",
        22: "Permanent crops",
        23: "Pastures",
        31: "Forests",
        5: "Surface water",
        32: "Shrub",
        33: "Open spaces",
        41: "Wetlands",
        13: "Mine dump",
        14: "Artificial vegetation",
        11: "Urban fabric",
        12: "Buildings"
    }

    # Map back to original codes
    mapped = np.zeros_like(pred, dtype=np.uint8)
    for net_class, orig_class in reverse_map.items():
        mapped[pred == net_class] = orig_class

    # Compute class percentages
    total_pixels = mapped.size
    summary = []
    for orig_code, desc in category_desc.items():
        count = np.sum(mapped == orig_code)
        if count > 0:
            pct = (count / total_pixels) * 100
            summary.append({
                "orig_code": orig_code,
                "description": desc,
                "pixel_count": int(count),
                "percentage": pct
            })

    return mapped, summary



def visualize_mask(mask, palette):
    color_mask = Image.fromarray(mask, mode="P")
    color_mask.putpalette(palette)

    plt.figure(figsize=(8, 8))
    plt.imshow(color_mask)
    plt.axis('off')
    plt.title("Prediction Visualization")
    plt.show()


def main(image_path):
    # ----- Paths -----
    weights_path = "checkpoint/model_multi_0.pth"
    palette_path = "checkpoint/palette.json"

    # ----- Checks -----
    assert os.path.exists(image_path), f"Image not found: {image_path}"
    assert os.path.exists(weights_path), f"Model weights not found: {weights_path}"
    assert os.path.exists(palette_path), f"Palette file not found: {palette_path}"

    with open(palette_path, "r") as f:
        palette_dict = json.load(f)
        palette = [v for sublist in palette_dict.values() for v in sublist]

    # ----- Device & Model -----
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = vit_base_patch8(num_classes=13)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=False)['model'])
    model.to(device)
    model.eval()

    # ----- Load and preprocess -----
    img = open_image(image_path)
    input_tensor = preprocess_image(img).to(device)
    print(input_tensor.shape)

    # ----- Inference -----
    with torch.no_grad():
        t0 = time_synchronized()
        output = model(input_tensor)
        t1 = time_synchronized()
        print(f"Inference time: {t1 - t0:.3f} seconds")

        prediction = output['out'].argmax(1).squeeze(0).cpu().numpy().astype(np.uint8)
        print(prediction.shape)
        print("Unique values in prediction:", np.unique(prediction))
     
    
    mapped_pred, summary = map_prediction_to_original(prediction)

    print("\nClass distribution detected:")
    for item in summary:
        print(f"{item['description']}: {item['percentage']:.2f}%")
       

    # ----- Visualize -----
    visualize_mask(prediction, palette)


if __name__ == '__main__':
    # Example usage
    test_image_path = "SegMunich_train_img_10.tif"  # Replace with your test image
    main(test_image_path)
    print('done')