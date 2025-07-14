import argparse
import json

from cv2 import transform
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, classification_report
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from dataset import VTDs
from model import MultiMobile

def test(model, dataloader, device, threshold, class_name):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for image, label in dataloader:
            image, label = image.to(device), label.to(device)
            
            pred = model(image)
            pred = pred.cpu().numpy()
            pred = np.where(pred >= threshold, 1, 0)
            
            all_preds.append(pred)
            all_labels.append(label)

    all_preds = np.stack(all_preds)
    all_labels = np.stack(all_labels)

    accuracy = accuracy_score(all_labels, all_preds)
    macro_precision = precision_score(all_labels, all_preds, average='macro')
    macro_recall = recall_score(all_labels, all_preds, average='macro')
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    micro_precision = precision_score(all_labels, all_preds, average='micro')
    micro_recall = recall_score(all_labels, all_preds, average='micro')
    micro_f1 = f1_score(all_labels, all_preds, average='micro')
    hamming = hamming_loss(all_labels, all_preds)

    metrics = {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'micro_precision': micro_precision,
        'micro_recall': micro_recall,
        'micro_f1': micro_f1,
        'hamming_loss': hamming
    }
    
    report = classification_report(all_labels, all_preds, target_names=class_name, zero_division=0, output_dict=True)

    return metrics, report

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test MultiMobile Model")

    parser.add_argument('--weight', type=str, default='model.pth', help='Path to save the model weights')
    parser.add_argument('--data', type=str, default='./data.json', help='Path to the dataset file')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--report', type=str, default='report.json', help='Path to save the classification report')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')

    args = parser.parse_args()

    threshold = args.threshold
    weight_path = args.weight
    data_path = args.data
    report_path = args.report
    config_path = args.config

    with open(data_path, 'r') as f:
        data = json.load(f)

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    device = config["device"].lower()
    test_batch_size = config["test_batch_size"]
    output_dim = config["output_dim"]

    model = MultiMobile(output_dim=output_dim)
    if weight_path:
        try:
            model.load_state_dict(torch.load(weight_path))
            print(f"Loaded model weights from {weight_path}")
        except FileNotFoundError:
            print(f"Model weights file {weight_path} not found. Starting with uninitialized model.")
    else:
        print("No model weights provided. Starting with uninitialized model.")
    model.to(device)
    
    transform = v2.Compose([
        v2.Resize((224, 224)),
        v2.ToDtype(dtype=torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = VTDs(data['test'], data['label'], data['classes'], transform=transform)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    with open(data['classes'], 'r') as f:
        class_name = json.load(f)['classes']

    metrics, report  = test(model, test_dataloader, device, threshold, class_name)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro Precision: {metrics['macro_precision']:.4f}")
    print(f"Macro Recall: {metrics['macro_recall']:.4f}")
    print(f"Macro F1 Score: {metrics['macro_f1']:.4f}")
    print(f"Micro Precision: {metrics['micro_precision']:.4f}")
    print(f"Micro Recall: {metrics['micro_recall']:.4f}")
    print(f"Micro F1 Score: {metrics['micro_f1']:.4f}")
    print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
    
    with open(report_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        json.dump(report, f, indent=4)