import argparse
import json

from matplotlib.font_manager import json_dump
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, hamming_loss, classification_report
import torch
from torch.utils.data import DataLoader

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

    parser.add_argument('--device', type=str, default='cuda', help='Device to run the model on (e.g., "cpu" or "cuda")')
    parser.add_argument('--weight', type=str, default='model.pth', help='Path to save the model weights')
    parser.add_argument('--data', type=str, default='./data.json', help='Path to the dataset file')
    parser.add_argument('--report', type=str, default='report.json', help='Path to save the classification report')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classification')
    parser.add_argument('--test_batch_size', type=int, default=32, help='Batch size for testing')
    parser.add_argument('--output_dim', type=int, default=1000, help='Output dimension for the classifier')

    args = parser.parse_args()
    
    device = args.device
    threshold = args.threshold
    weight_path = args.weight
    data_path = args.data
    test_batch_size = args.test_batch_size
    output_dim = args.output_dim
    report_path = args.report

    with open(data_path, 'r') as f:
        data = json.load(f)
        test_data_path = data["test"]
        class_name = data["class"]

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
    
    test_dataset = VTDs()
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)

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