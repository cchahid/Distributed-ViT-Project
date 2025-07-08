
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import timm
from tqdm import tqdm
import time

# --- Configuration ---
DATA_DIR = './tiny-imagenet-200'
NUM_CLASSES = 200
BATCH_SIZE = 64 # Using a safe batch size
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"Running on device: {DEVICE}")

    # --- Data Loading ---
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{DATA_DIR}/val", transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # --- Model, Loss, and Optimizer ---
    print("Loading a smaller Vision Transformer model...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    ## UPDATE: Using the new torch.amp API
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE == 'cuda'))

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        start_time = time.time()
        total_samples = 0

        # Training Phase
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            ## UPDATE: Using the new torch.amp API
            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_samples += images.size(0)

        end_time = time.time()
        epoch_duration = end_time - start_time
        throughput = total_samples / epoch_duration

        print(f"\nEpoch {epoch+1} Training Complete.")
        print(f"  - Time: {epoch_duration:.2f} seconds")
        print(f"  - Throughput: {throughput:.2f} images/sec")
        
        # Validation Loop
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        accuracy = 100 * correct / total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"  - Validation Loss: {avg_val_loss:.4f}")
        print(f"  - Validation Accuracy: {accuracy:.2f} % \n")

if __name__ == '__main__':
    main()
