import modal
import os
from pathlib import Path

# Define the Modal app
app = modal.App("simple-dnn-training")

# Create a custom image with all required dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install([
        "torch>=2.0.0",
        "torchvision",
        "transformers>=4.20.0",
        "datasets>=2.0.0",
        "scikit-learn",
        "numpy",
        "huggingface-hub",
    ])
)

# Create a persistent volume for storing models and data
volume = modal.Volume.from_name("dnn-experiments", create_if_missing=True)

# Mount the volume at /data
VOLUME_PATH = "/data"


@app.function(
    image=image,
    gpu="T4",  # Small GPU for experimentation
    volumes={VOLUME_PATH: volume},
    timeout=3600,  # 1 hour timeout
    secrets=[modal.Secret.from_name("huggingface-secret")],  # Optional: for private datasets
)
def train_dnn_remote(
    dataset_name: str = "imdb",
    model_type: str = "text", 
    num_epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    experiment_name: str = None
):
    """
    Train a simple DNN on Modal Labs with GPU support
    
    Args:
        dataset_name: Dataset to use ("imdb", "cifar10", "mnist")
        model_type: Model architecture ("text" or "simple")
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        experiment_name: Name for the experiment (auto-generated if None)
    """
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split
    from datasets import load_dataset
    from transformers import AutoTokenizer
    import numpy as np
    from sklearn.metrics import accuracy_score, classification_report
    import time
    import json
    from datetime import datetime
    
    # Import our model classes (we'll need to copy the code since we can't import from local files)
    # Model definitions (copied from model.py)
    class SimpleDNN(nn.Module):
        def __init__(self, input_dim, hidden_dims, output_dim, dropout_rate=0.2):
            super(SimpleDNN, self).__init__()
            
            self.layers = nn.ModuleList()
            
            # Input layer
            prev_dim = input_dim
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
            
            # Output layer
            self.output_layer = nn.Linear(prev_dim, output_dim)
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            # Flatten input if needed (for image data)
            if len(x.shape) > 2:
                x = x.view(x.size(0), -1)
                
            for layer in self.layers:
                x = torch.relu(layer(x))
                x = self.dropout(x)
                
            x = self.output_layer(x)
            return x

    class TextClassificationDNN(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dims, num_classes, dropout_rate=0.2):
            super(TextClassificationDNN, self).__init__()
            
            self.embedding = nn.Embedding(vocab_size, embedding_dim)
            self.embedding_dropout = nn.Dropout(dropout_rate)
            
            self.layers = nn.ModuleList()
            prev_dim = embedding_dim
            
            for hidden_dim in hidden_dims:
                self.layers.append(nn.Linear(prev_dim, hidden_dim))
                prev_dim = hidden_dim
                
            self.output_layer = nn.Linear(prev_dim, num_classes)
            self.dropout = nn.Dropout(dropout_rate)
            
        def forward(self, x):
            # x shape: (batch_size, sequence_length)
            x = self.embedding(x)  # (batch_size, sequence_length, embedding_dim)
            x = self.embedding_dropout(x)
            
            # Global average pooling
            x = torch.mean(x, dim=1)  # (batch_size, embedding_dim)
            
            for layer in self.layers:
                x = torch.relu(layer(x))
                x = self.dropout(x)
                
            x = self.output_layer(x)
            return x
    
    # Set experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"{dataset_name}_{model_type}_dnn_{timestamp}"
    
    print(f"Starting experiment: {experiment_name}")
    print(f"Dataset: {dataset_name}")
    print(f"Model type: {model_type}")
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data preparation functions
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
    
    # Load and prepare data
    print("Loading dataset...")
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        
        train_dataset = prepare_text_data(dataset["train"], tokenizer, max_length=512)
        test_dataset = prepare_text_data(dataset["test"], tokenizer, max_length=512)
        
        # Model config for text
        vocab_size = tokenizer.vocab_size
        embedding_dim = 128
        hidden_dims = [256, 128]
        num_classes = 2
        
        model = TextClassificationDNN(vocab_size, embedding_dim, hidden_dims, num_classes)
        
    elif dataset_name == "cifar10":
        dataset = load_dataset("cifar10")
        train_dataset = prepare_image_data(dataset["train"])
        test_dataset = prepare_image_data(dataset["test"])
        
        # Model config for CIFAR-10
        input_dim = 32 * 32 * 3
        hidden_dims = [512, 256, 128]
        output_dim = 10
        
        model = SimpleDNN(input_dim, hidden_dims, output_dim)
        
    elif dataset_name == "mnist":
        dataset = load_dataset("mnist")
        train_dataset = prepare_image_data(dataset["train"])
        test_dataset = prepare_image_data(dataset["test"])
        
        # Model config for MNIST
        input_dim = 28 * 28
        hidden_dims = [512, 256]
        output_dim = 10
        
        model = SimpleDNN(input_dim, hidden_dims, output_dim)
    
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Create validation split (10% of training data)
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Setup model, loss, and optimizer
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training functions
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
    
    # Training loop
    print("\nStarting training...")
    best_val_accuracy = 0
    training_history = []
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_accuracy = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss, val_accuracy, _, _ = evaluate(model, val_loader, criterion, device)
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path = f"{VOLUME_PATH}/{experiment_name}_best.pt"
            torch.save(model.state_dict(), model_path)
        
        epoch_time = time.time() - start_time
        
        epoch_results = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "epoch_time": epoch_time
        }
        training_history.append(epoch_results)
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}")
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
    
    # Save final model and results
    final_model_path = f"{VOLUME_PATH}/{experiment_name}_final.pt"
    torch.save(model.state_dict(), final_model_path)
    
    # Save training history and results
    results = {
        "experiment_name": experiment_name,
        "dataset_name": dataset_name,
        "model_type": model_type,
        "hyperparameters": {
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
        },
        "training_history": training_history,
        "final_test_accuracy": test_accuracy,
        "final_test_loss": test_loss,
        "best_val_accuracy": best_val_accuracy,
        "model_parameters": sum(p.numel() for p in model.parameters()),
    }
    
    results_path = f"{VOLUME_PATH}/{experiment_name}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nExperiment completed!")
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Results saved to: {results_path}")
    print(f"Models saved to: {VOLUME_PATH}")
    
    # Commit volume changes to persist the models and results
    volume.commit()
    
    return results


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
)
def list_experiments():
    """List all experiment results stored in the volume"""
    import json
    import os
    
    results_files = [f for f in os.listdir(VOLUME_PATH) if f.endswith("_results.json")]
    
    if not results_files:
        print("No experiments found.")
        return []
    
    experiments = []
    for results_file in results_files:
        with open(f"{VOLUME_PATH}/{results_file}", "r") as f:
            results = json.load(f)
            experiments.append({
                "name": results["experiment_name"],
                "dataset": results["dataset_name"],
                "test_accuracy": results["final_test_accuracy"],
                "best_val_accuracy": results["best_val_accuracy"],
            })
    
    print("Experiments:")
    for exp in experiments:
        print(f"  {exp['name']}: {exp['dataset']} - Test Acc: {exp['test_accuracy']:.4f}")
    
    return experiments


@app.local_entrypoint()
def main(
    dataset: str = "imdb",
    model_type: str = "text",
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    experiment_name: str = None,
    list_only: bool = False,
):
    """
    Main entrypoint for training DNNs on Modal Labs
    
    Examples:
        modal run modal_train.py                                    # Train IMDB text classification
        modal run modal_train.py --dataset cifar10 --model_type simple --epochs 10
        modal run modal_train.py --list_only                       # List all experiments
    """
    if list_only:
        list_experiments.remote()
    else:
        results = train_dnn_remote.remote(
            dataset_name=dataset,
            model_type=model_type,
            num_epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            experiment_name=experiment_name,
        )
        print("\nExperiment completed successfully!")
        return results


# Convenience functions for different datasets
@app.local_entrypoint()
def train_imdb():
    """Train IMDB sentiment classification"""
    return train_dnn_remote.remote(dataset_name="imdb", model_type="text", num_epochs=5)


@app.local_entrypoint() 
def train_cifar10():
    """Train CIFAR-10 image classification"""
    return train_dnn_remote.remote(dataset_name="cifar10", model_type="simple", num_epochs=10)


@app.local_entrypoint()
def train_mnist():
    """Train MNIST digit classification"""
    return train_dnn_remote.remote(dataset_name="mnist", model_type="simple", num_epochs=5)