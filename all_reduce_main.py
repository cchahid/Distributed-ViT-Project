
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
import timm
from tqdm import tqdm
import time
import horovod.torch as hvd
import os

# --- Configuration ---
DATA_DIR = './tiny-imagenet-200'
NUM_CLASSES = 200
BATCH_SIZE = 64  # This will be the batch size PER GPU
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = 'cuda'

def main():
    # HVD UPDATE: Initialize Horovod
    hvd.init()
    # HVD UPDATE: Pin each process to a GPU. This is crucial.
    torch.cuda.set_device(hvd.local_rank())
    # HVD UPDATE: Only print logs on the main process (rank 0) to avoid clutter
    is_main_process = hvd.rank() == 0

    if is_main_process:
        print(f"Running All-Reduce training on {hvd.size()} GPUs.")

    # --- Data Loading ---
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
    val_dataset = ImageFolder(root=f"{DATA_DIR}/val", transform=transform)
    
    # HVD UPDATE: Use DistributedSampler to partition the data across GPUs
    train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    val_sampler = DistributedSampler(val_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    
    # HVD UPDATE: The 'shuffle' argument must be False when using a sampler
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, num_workers=2, pin_memory=True)

    # --- Model, Loss, and Optimizer ---
    if is_main_process:
        print("Loading Vision Transformer model...")
    model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
    model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    # HVD UPDATE: Scale the learning rate by the number of GPUs. This is a best practice.
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE * hvd.size())

    # HVD UPDATE: Wrap the optimizer with Horovod's DistributedOptimizer
    optimizer = hvd.DistributedOptimizer(optimizer, named_parameters=model.named_parameters())
    
    # HVD UPDATE: Broadcast initial parameters from rank 0 to all other processes
    # This ensures all models start with the same weights.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    scaler = torch.amp.GradScaler(enabled=(DEVICE == 'cuda'))

    # --- Training Loop ---
    if is_main_process:
        print("Starting training...")
    for epoch in range(EPOCHS):
        model.train()
        # HVD UPDATE: Set the epoch for the sampler to ensure proper shuffling each epoch
        train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        
        # Training Phase
        # HVD UPDATE: Disable the progress bar on non-main processes
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]", disable=not is_main_process)
        for images, labels in train_iterator:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        
        # Validation Loop (run on all workers to get aggregated results)
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            # HVD UPDATE: Disable progress bar on non-main processes
            val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]", disable=not is_main_process)
            for images, labels in val_iterator:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        # HVD UPDATE: Aggregate metrics from all processes using allreduce
        avg_val_loss = hvd.allreduce(torch.tensor(val_loss), name='avg_loss').item() / len(val_loader)
        total_correct = hvd.allreduce(torch.tensor(correct), name='sum_correct').item()
        total_samples_val = hvd.allreduce(torch.tensor(total), name='sum_total').item()
        accuracy = 100 * total_correct / total_samples_val
        
        # HVD UPDATE: Only the main process prints the final results
        if is_main_process:
            end_time = time.time()
            epoch_duration = end_time - start_time
            # Throughput is calculated based on the total number of samples processed across all GPUs
            global_throughput = (len(train_sampler) * hvd.size()) / epoch_duration

            print(f"\nEpoch {epoch+1} Training Complete.")
            print(f"  - Time: {epoch_duration:.2f} seconds")
            print(f"  - Global Throughput: {global_throughput:.2f} images/sec")
            print(f"  - Aggregated Validation Loss: {avg_val_loss:.4f}")
            print(f"  - Aggregated Validation Accuracy: {accuracy:.2f} % \n")

if __name__ == '__main__':
    main()

