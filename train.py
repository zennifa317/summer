import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultiMobile
from dataset import VTDs
from utils import create_versioned_dir

def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    ave_loss = 0.0

    for image, labels in dataloader:
        image, labels = image.to(device), labels.to(device)
        
        optimizer.zero_grad()

        pred = model(image)
        loss = criterion(pred, labels)
        loss.backward()

        optimizer.step()
        total_loss += loss.item()
        
    ave_loss = total_loss / len(dataloader)

    return ave_loss

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    ave_loss = 0.0

    with torch.no_grad():
        for image, labels in dataloader:
            image, labels = image.to(device), labels.to(device)

            pred = model(image)
            loss = criterion(pred, labels)
            total_loss += loss.item()
        
        ave_loss = total_loss / len(dataloader)

    return ave_loss

def plot_losses(train_losses, valid_losses, epochs):
    x = range(1, epochs + 1)

    plt.figure(figsize=(10, 5))

    plt.plot(x, train_losses, label='Train Loss', color='blue')
    plt.plot(x, valid_losses, label='Validation Loss', color='orange')
    
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Losses')
    
    plt.legend()

    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train MultiMobile Model")

    parser.add_argument('--weight', type=str, default=None, help='Path to fine-tuned model weights')
    parser.add_argument('--config', type=str, default='config.json', help='Path to the config file')
    parser.add_argument('--data', type=str, default='./data/data.json', help='Path to the dataset file')
    parser.add_argument('--output', type=str, default='exp', help='Path to the output folder')

    args = parser.parse_args()

    weight_path = args.weight
    config_path = args.config
    data_path = args.data
    output_name = args.output

    if not os.path.exists('./train'):
        os.mkdir('./train')

    output_path = create_versioned_dir(base_name=output_name, dir="./train")

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    with open(config_path, 'r') as f:
        config = json.load(f)

    output_dim = config["output_dim"]
    epochs = config["epochs"]
    device = config["device"].lower()
    train_batch_size = config["train_batch_size"]
    valid_batch_size = config["valid_batch_size"]
    learning_rate = config["learning_rate"]
    optimizer_type = config["optimizer"].lower()
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]

    with open(data_path, 'r') as f:
        data = json.load(f)

    print(f'Using device: {device}')

    model = MultiMobile(output_dim=output_dim)
    if weight_path is not None:
        try:
            model.load_state_dict(torch.load(weight_path))
            print(f"Loaded model weights from {weight_path}")
        except FileNotFoundError:
            print(f"Model weights file {weight_path} not found. Starting with uninitialized model.")
    else:
        print("No model weights provided. Starting with uninitialized model.")
    model.to(device)

    criterion = nn.BCELoss()

    if optimizer_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_type == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=0.0)

    train_dataset = VTDs(data['train'], data['label'], data['classes'], transform=None)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True)

    valid_dataset = VTDs(data['valid'], data['label'], data['classes'], transform=None)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False)

    train_losses = []
    valid_losses = []

    for epoch in tqdm(range(epochs)):
        train_loss = train(model, train_dataloader, criterion, optimizer, device)
        valid_loss = validate(model, valid_dataloader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')

    plot_losses(train_losses, valid_losses, epochs)
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved to {weight_path}")