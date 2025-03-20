import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
from tqdm import tqdm
from models.bcm_layers import FastBlockCirculantLinear

def train_model(model, train_dataloader, test_dataloader, num_epochs=5, lr=0.0001, device='cuda'):
    # Device setup
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_accuracy = 0.0
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for texts, masks, labels in progress_bar:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(texts, masks)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Evaluate
        accuracy = evaluate_model(model, test_dataloader, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Test Accuracy: {accuracy:.2f}%")
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            # Save best model
            torch.save(model.state_dict(), 'best_bcm_transformer.pt')
    
    return model

def evaluate_model(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for texts, masks, labels in dataloader:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            outputs = model(texts, masks)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def analyze_compression(model):
    # Count BCM parameters
    bcm_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate standard model equivalent parameters
    standard_params = 0
    bcm_params_breakdown = 0
    
    for name, module in model.named_modules():
        if isinstance(module, FastBlockCirculantLinear):
            # Full matrix would have in_features * out_features parameters
            full_params = module.in_features * module.out_features
            # BCM version has blocks * block_size parameters + bias
            bcm_params_layer = module.n_blocks_rows * module.n_blocks_cols * module.block_size + module.out_features
            
            standard_params += full_params + module.out_features  # adding bias
            bcm_params_breakdown += bcm_params_layer
            
            print(f"Layer {name}:")
            print(f"  - Standard params: {full_params + module.out_features:,}")
            print(f"  - BCM params: {bcm_params_layer:,}")
            print(f"  - Compression ratio: {(full_params + module.out_features) / bcm_params_layer:.2f}x")
    
    # Calculate overall compression ratio
    compression_ratio = standard_params / bcm_params_breakdown if bcm_params_breakdown > 0 else 0
    
    print("\nOverall Compression:")
    print(f"Standard model parameters (theoretical): {standard_params:,}")
    print(f"BCM model parameters: {bcm_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")
    
    return compression_ratio
