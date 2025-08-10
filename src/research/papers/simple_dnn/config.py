from dataclasses import dataclass
from typing import List


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    seed: int = 42
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"
    save_model: bool = True
    model_save_path: str = "models/"


@dataclass
class SimpleDNNConfig:
    input_dim: int = 784  # For MNIST-like data
    hidden_dims: List[int] = None
    output_dim: int = 10
    dropout_rate: float = 0.2
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]


@dataclass
class TextDNNConfig:
    vocab_size: int = 50000
    embedding_dim: int = 128
    hidden_dims: List[int] = None
    num_classes: int = 2
    dropout_rate: float = 0.2
    max_length: int = 512
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128]


@dataclass
class DataConfig:
    dataset_name: str = "imdb"  # "imdb", "cifar10", or "mnist"
    data_dir: str = "./data"
    num_workers: int = 2
    train_split: str = "train"
    test_split: str = "test"
    validation_split: float = 0.1


@dataclass
class ExperimentConfig:
    experiment_name: str = "simple_dnn_experiment"
    model_type: str = "text"  # "simple" or "text"
    training: TrainingConfig = None
    model: SimpleDNNConfig | TextDNNConfig = None
    data: DataConfig = None
    
    def __post_init__(self):
        if self.training is None:
            self.training = TrainingConfig()
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            if self.model_type == "text":
                self.model = TextDNNConfig()
            else:
                self.model = SimpleDNNConfig()


# Predefined configurations for different experiments
def get_imdb_config():
    config = ExperimentConfig(
        experiment_name="imdb_sentiment_classification",
        model_type="text"
    )
    config.data.dataset_name = "imdb"
    config.model.num_classes = 2
    return config


def get_cifar10_config():
    config = ExperimentConfig(
        experiment_name="cifar10_classification",
        model_type="simple"
    )
    config.data.dataset_name = "cifar10"
    config.model.input_dim = 32 * 32 * 3
    config.model.output_dim = 10
    config.model.hidden_dims = [512, 256, 128]
    return config


def get_mnist_config():
    config = ExperimentConfig(
        experiment_name="mnist_classification",
        model_type="simple"
    )
    config.data.dataset_name = "mnist"
    config.model.input_dim = 28 * 28
    config.model.output_dim = 10
    config.model.hidden_dims = [512, 256]
    return config