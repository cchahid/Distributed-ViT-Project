
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
BATCH_SIZE = 64  # This will be the batch size PER WORKER
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = 'cuda'

def main():
    # --- Horovod Initialization ---
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())

    # PS UPDATE: Assign roles. Rank 0 is the parameter server. All others are workers.
    is_server = hvd.rank() == 0
    is_worker = not is_server

    if is_server:
        print(f"Starting Parameter Server training with 1 Server and {hvd.size() - 1} Workers.")
        # The server needs to know the number of workers for averaging gradients
        num_workers = hvd.size() - 1
    
    # --- Model, Loss, and Optimizer ---
    # PS UPDATE: The server holds the master model, optimizer, and scaler.
    if is_server:
        print("Server: Loading Vision Transformer model...")
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
        model.to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scaler = torch.amp.GradScaler(enabled=(DEVICE == 'cuda'))
    else:
        # PS UPDATE: Workers only need the model structure. They don't optimize.
        model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=NUM_CLASSES)
        model.to(DEVICE)

    # PS UPDATE: The server broadcasts the initial model parameters to all workers.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    
    # PS UPDATE: Only workers need the loss function for the backward pass.
    criterion = nn.CrossEntropyLoss()

    # --- Data Loading (Worker-side only) ---
    if is_worker:
        transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        train_dataset = ImageFolder(root=f"{DATA_DIR}/train", transform=transform)
        # PS UPDATE: The sampler is configured for workers only.
        # num_replicas is the number of workers, and rank is adjusted.
        train_sampler = DistributedSampler(train_dataset, num_replicas=hvd.size() - 1, rank=hvd.rank() - 1)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=2, pin_memory=True)

    # --- Training Loop ---
    if is_server:
        print("Server: Ready to receive gradients...")

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        if is_worker:
            model.train()
            train_sampler.set_epoch(epoch)
            # PS UPDATE: Workers iterate through their data shard.
            # The progress bar is now managed by the worker rank.
            train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Worker {hvd.rank()}]", leave=False)
            for images, labels in train_iterator:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                
                # 1. WORKER: Compute gradients
                model.zero_grad()
                with torch.amp.autocast(device_type=DEVICE, enabled=(DEVICE == 'cuda')):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                loss.backward()

                # 2. WORKER: Send gradients to the server (rank 0)
                for param in model.parameters():
                    if param.grad is not None:
                        # Use allreduce which acts as send-to-root-and-sum here
                        hvd.allreduce(param.grad, op=hvd.Sum, name=f'grad_{id(param)}')
                
                # 4. WORKER: Receive updated parameters from the server
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        # PS UPDATE: This is the server's main role during the training steps.
        if is_server:
            # This is a simplified synchronous loop. The server effectively waits
            # for each batch to be processed by the workers.
            num_batches = len(train_dataset) // (num_workers * BATCH_SIZE) if is_worker else 0
            
            server_iterator = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{EPOCHS} [Server Updating]", leave=False)
            for _ in server_iterator:
                # 3. SERVER: Average gradients and update the master model
                # The gradients are already summed via allreduce. We average them.
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= num_workers
                
                # The scaler is used here, on the server, before the optimizer step
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

                # 4. SERVER: Broadcast the updated parameters back to all workers
                hvd.broadcast_parameters(model.state_dict(), root_rank=0)

        if is_server:
            end_time = time.time()
            epoch_duration = end_time - start_time
            print(f"\nEpoch {epoch+1} Complete.")
            print(f"  - Time: {epoch_duration:.2f} seconds")
            # Note: A direct throughput comparison is complex here because the server
            # and workers have different workloads.

    if is_server:
        print("Training finished.")

if __name__ == '__main__':
    main()
