# Simple Deep Neural Network Implementation

This directory contains a simple deep neural network implementation using PyTorch with support for both local training and remote execution on Modal Labs GPU infrastructure.

## Overview

The implementation provides:
- **Flexible DNN architectures**: Support for both simple MLPs and text classification models
- **Multiple datasets**: IMDB sentiment analysis, CIFAR-10, and MNIST
- **Local training**: Full training pipeline with validation and evaluation
- **Remote GPU training**: Modal Labs integration for cloud-based training on T4 GPUs

## Files

- `model.py` - Neural network architectures (SimpleDNN, TextClassificationDNN)
- `config.py` - Configuration classes and predefined experiment configs
- `train.py` - Local training script with full evaluation pipeline
- `modal_train.py` - Modal Labs remote training with GPU support
- `README.md` - This documentation

## Quick Start

### Local Training

```bash
# Train IMDB sentiment classification locally
cd src/research/papers/simple_dnn
python train.py

# The script will automatically download the IMDB dataset and train a text classification model
```

### Remote Training on Modal Labs

First, make sure you have Modal CLI installed and authenticated:

```bash
pip install modal
modal token new
```

Then run training on Modal with GPU:

```bash
# Train IMDB sentiment classification on T4 GPU
modal run modal_train.py

# Train CIFAR-10 with custom parameters
modal run modal_train.py --dataset cifar10 --model_type simple --epochs 10

# Train MNIST
modal run modal_train.py --dataset mnist --model_type simple --epochs 5

# List all completed experiments
modal run modal_train.py --list_only
```

### Convenience Commands

```bash
# Pre-configured dataset training
modal run modal_train.py::train_imdb
modal run modal_train.py::train_cifar10  
modal run modal_train.py::train_mnist
```

## Model Architectures

### SimpleDNN
- Multi-layer perceptron for image classification
- Configurable hidden layer dimensions
- Dropout regularization
- ReLU activations

### TextClassificationDNN  
- Embedding layer for text tokenization
- Global average pooling over sequence dimension
- Multi-layer classification head
- Suitable for sentiment analysis and text classification

## Datasets Supported

1. **IMDB Movie Reviews** (`dataset="imdb"`)
   - Binary sentiment classification
   - 50,000 movie reviews
   - Automatic tokenization with BERT tokenizer

2. **CIFAR-10** (`dataset="cifar10"`)
   - 10-class image classification  
   - 32x32 RGB images
   - 60,000 samples

3. **MNIST** (`dataset="mnist"`)
   - 10-class digit recognition
   - 28x28 grayscale images
   - 70,000 samples

## Configuration

The `config.py` file provides flexible configuration through dataclasses:

```python
from config import get_imdb_config, get_cifar10_config, get_mnist_config

# Get pre-configured settings
config = get_imdb_config()
config.training.num_epochs = 10
config.training.batch_size = 64
```

## Model Persistence

### Local Training
Models are saved to `models/` directory:
- `{experiment_name}_best.pt` - Best validation accuracy model
- `{experiment_name}_final.pt` - Final trained model

### Modal Training  
Models and results are saved to persistent Modal volume:
- Automatic experiment tracking
- JSON results with training history
- Best and final model checkpoints

## Example Results

Typical performance after 5 epochs:

- **IMDB**: ~85-90% test accuracy
- **CIFAR-10**: ~60-70% test accuracy  
- **MNIST**: ~95-98% test accuracy

## Extending the Implementation

To add new datasets or architectures:

1. Add dataset loading logic in `train.py` and `modal_train.py`
2. Create new model classes in `model.py` 
3. Add configuration classes in `config.py`
4. Update the `create_model()` factory function

This implementation serves as a foundation for experimenting with deep learning concepts and can be easily extended for more complex architectures and datasets.