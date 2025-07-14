import argparse
import json
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, default_collate
from torchvision.transforms import v2
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import MultiMobile
from dataset import VTDs
from utils import create_versioned_dir

def train(model, dataloader, criterion, optimizer, device, num_classes):
    model.train()
    total_loss = 0.0
    ave_loss = 0.0

    cutmix = v2.CutMix(num_classes=num_classes)
    mixup = v2.MixUp(num_classes=num_classes)
    cutmix_or_mixup = v2.RandomChoice([cutmix, mixup])

    for images, labels in dataloader:
        images, labels = cutmix_or_mixup(images, labels)
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        preds = model(images)
        loss = criterion(preds, labels)
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
    parser.add_argument('--check_point', type=str, default=None, help='Path to the checkpoint file')

    args = parser.parse_args()

    weight_path = args.weight
    config_path = args.config
    data_path = args.data
    output_name = args.output
    checkpoint_path = args.check_point

    output_path = create_versioned_dir(base_name=output_name, dir="./train")
    result_weight_dir = os.path.join(output_path, 'weights')

    os.makedirs(result_weight_dir, exist_ok=True)

    with open(config_path, 'r') as f:
        config = json.load(f)
        
    with open(output_path, 'w') as f:
        json.dump(config, f)

    with open(data_path, 'r') as f:
        data = json.load(f)

    output_dim = config["output_dim"]
    epochs = config["epochs"]
    device = config["device"].lower()
    train_batch_size = config["train_batch_size"]
    valid_batch_size = config["valid_batch_size"]
    learning_rate = config["learning_rate"]
    optimizer_type = config["optimizer"].lower()
    momentum = config["momentum"]
    weight_decay = config["weight_decay"]

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
    
    train_transform = v2.Compose([
    v2.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    v2.RandomRotation(degrees=(-15, 15)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    valid_transform = v2.Compose([
    v2.Resize(size=(224, 224), antialias=True),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = VTDs(data['train'], data['label'], data['classes'], transform=train_transform)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=2)

    valid_dataset = VTDs(data['valid'], data['label'], data['classes'], transform=valid_transform)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=valid_batch_size, shuffle=False, num_workers=2)
    
    start_epoch = 0
    best_valid_loss = float('inf')
    train_losses = []
    valid_losses = []
    
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_valid_loss = checkpoint['best_valid_loss']
        train_losses = checkpoint['train_losses']
        valid_losses = checkpoint['valid_losses']
        config = checkpoint['config']
        print(f"Resumed from epoch {start_epoch}.")


    for epoch in tqdm(range(start_epoch, epochs)):
        train_loss = train(model, train_dataloader, criterion, optimizer, device, output_dim)
        valid_loss = validate(model, valid_dataloader, criterion, device)
        scheduler.step()

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Valid Loss: {valid_loss:.4f}')
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), os.path.join(result_weight_dir, 'best_model.pth'))
        
        if (epoch + 1) % 25 == 0:
            checkpoint_path = os.path.join(result_weight_dir, f'{epoch+1}epoch_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_valid_loss': best_valid_loss,
                'train_losses': train_losses,
                'valid_losses': valid_losses
                }, checkpoint_path)
            print(f"Epoch {epoch+1}, Checkpoint saved.")

    plot_losses(train_losses, valid_losses, epochs)
    torch.save(model.state_dict(), weight_path)
    print(f"Model weights saved to {weight_path}")