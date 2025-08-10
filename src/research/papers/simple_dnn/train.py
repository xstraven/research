import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from datasets import load_dataset
from transformers import AutoTokenizer
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import time
from pathlib import Path

from model import create_model
from config import ExperimentConfig, get_imdb_config, get_cifar10_config, get_mnist_config


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_preference="auto"):
    if device_preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_preference)


def prepare_text_data(dataset, tokenizer, max_length=512):
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=max_length,
            return_tensors="pt"
        )
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "label"])
    return tokenized_dataset


def prepare_image_data(dataset):
    def transform_function(examples):
        # Convert PIL images to tensors and normalize
        images = []
        for img in examples["image"]:
            # Convert to tensor and normalize
            img_array = np.array(img).astype(np.float32) / 255.0
            if len(img_array.shape) == 2:  # Grayscale
                img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            else:  # RGB
                img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            images.append(img_tensor)
        
        return {"image": images, "label": examples["label"]}
    
    dataset = dataset.map(transform_function, batched=True)
    dataset.set_format(type="torch", columns=["image", "label"])
    return dataset


def load_data(config: ExperimentConfig):
    data_config = config.data
    
    if data_config.dataset_name == "imdb":
        dataset = load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = prepare_text_data(
            dataset[data_config.train_split], 
            tokenizer, 
            max_length=config.model.max_length
        )
        test_dataset = prepare_text_data(
            dataset[data_config.test_split], 
            tokenizer, 
            max_length=config.model.max_length
        )
        
        # Update vocab size in config
        config.model.vocab_size = tokenizer.vocab_size
        
    elif data_config.dataset_name == "cifar10":
        dataset = load_dataset("cifar10")
        train_dataset = prepare_image_data(dataset[data_config.train_split])
        test_dataset = prepare_image_data(dataset[data_config.test_split])
        
    elif data_config.dataset_name == "mnist":
        dataset = load_dataset("mnist")
        train_dataset = prepare_image_data(dataset[data_config.train_split])
        test_dataset = prepare_image_data(dataset[data_config.test_split])
        
    else:
        raise ValueError(f"Unsupported dataset: {data_config.dataset_name}")
    
    # Create validation split
    if data_config.validation_split > 0:
        train_size = int((1 - data_config.validation_split) * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    else:
        val_dataset = None
    
    return train_dataset, val_dataset, test_dataset


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    for batch in train_loader:
        if "input_ids" in batch:  # Text data
            inputs = batch["input_ids"].to(device)
        else:  # Image data
            inputs = batch["image"].to(device)
        
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        predictions = torch.argmax(outputs, dim=1)
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return total_loss / len(train_loader), accuracy


def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            if "input_ids" in batch:  # Text data
                inputs = batch["input_ids"].to(device)
            else:  # Image data
                inputs = batch["image"].to(device)
            
            labels = batch["label"].to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_predictions)
    return total_loss / len(data_loader), accuracy, all_predictions, all_labels


def train(config: ExperimentConfig):
    print(f"Starting experiment: {config.experiment_name}")
    print(f"Model type: {config.model_type}")
    print(f"Dataset: {config.data.dataset_name}")
    
    # Set seed for reproducibility
    set_seed(config.training.seed)
    
    # Get device
    device = get_device(config.training.device)
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_dataset, val_dataset, test_dataset = load_data(config)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=True,
        num_workers=config.data.num_workers
    )
    
    if val_dataset:
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.training.batch_size, 
            shuffle=False,
            num_workers=config.data.num_workers
        )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=config.training.batch_size, 
        shuffle=False,
        num_workers=config.data.num_workers
    )
    
    print(f"Train samples: {len(train_dataset)}")
    if val_dataset:
        print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create model
    model = create_model(config.model_type, config.model)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_accuracy = 0
    
    for epoch in range(config.training.num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        if val_dataset:
            val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
            
            # Save best model
            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy
                if config.training.save_model:
                    save_path = Path(config.training.model_save_path)
                    save_path.mkdir(exist_ok=True)
                    torch.save(model.state_dict(), save_path / f"{config.experiment_name}_best.pt")
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{config.training.num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
        if val_dataset:
            print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        print(f"  Time: {epoch_time:.2f}s")
        print()
    
    # Final evaluation on test set
    print("Evaluating on test set...")
    test_loss, test_accuracy, test_predictions, test_labels = evaluate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions))
    
    # Save final model
    if config.training.save_model:
        save_path = Path(config.training.model_save_path)
        save_path.mkdir(exist_ok=True)
        torch.save(model.state_dict(), save_path / f"{config.experiment_name}_final.pt")
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # Example usage - you can modify this to run different experiments
    
    # IMDB sentiment classification
    config = get_imdb_config()
    train(config)
    
    # Uncomment to try other datasets:
    # config = get_cifar10_config()
    # train(config)
    
    # config = get_mnist_config()
    # train(config)