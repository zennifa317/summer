import argparse
import json

import torch
from torchvision.transforms import v2
from torchvision.io import decode_image
from PIL import Image, ImageDraw
import numpy as np

from model import MultiMobile

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Detection MultiMobile Model")

    parser.add_argument('--input_image', type=str, default='input.jpg', help='Path to the input image')
    parser.add_argument('--weight', type=str, default='model.pth', help='Path to the model weights')
    parser.add_argument('--output_dim', type=int, default=100, help='Output dimension for the classifier')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model on (e.g., "cpu" or "cuda")')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for detection')
    parser.add_argument('--data', type=str, default='data.json', help='Path to the dataset file')

    args = parser.parse_args()

    input_image = args.input_image
    weight = args.weight
    output_dim = args.output_dim
    device = args.device
    threshold = args.threshold
    data_path = args.data

    with open(data_path, 'r') as f:
        data = json.load(f)
        class_name = data["class"]

    model = MultiMobile(output_dim=output_dim)
    #model.load_state_dict(torch.load(weight))
    model.to(device)
    model.eval()
    
    image = decode_image(input_image)
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(image)
        pred = pred.cpu().numpy()

    indices = np.where(pred >= threshold)
    
    for index in indices[1]:
        class_name = class_name[index]
        score = pred[0, index]
        
        print(f"Class: {class_name}, Score: {score:.4f}")