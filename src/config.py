import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    device: str = "gpu" if torch.cuda.is_available() else "cpu"
    n_estimators: int = 10
    epochs: int = int(1e12)
    patience: int = 50
    lr: int = 5e-3
    batch_size: int = 64
    starting_model: int = 0
    sample_with_replacement: int = True
    random_architecture: dict = field(default_factory= lambda: {
        'enabled': True,
        'min_num_layers': 1,
        'max_num_layers': 2
    })
    random_num_of_features: dict = field(default_factory= lambda: {
        'enabled': False,
        'min_num_of_features': 20,
        'max_num_of_features': 30
    })
    dropout: dict = field(default_factory= lambda: {
        'enabled': False,
        'p': 0.2
    })

    # def __post_init__(self, dataset, seed, model_frac):
    #

    @property
    def random_num_of_features_enabled(self):
        return self.random_num_of_features['enabled']
    @property
    def min_num_of_features(self):
        return self.random_num_of_features['min_num_of_features']
    @property
    def max_num_of_features(self):
        return self.random_num_of_features['max_num_of_features']

    @property
    def random_architecture_enabled(self):
        return self.random_architecture['enabled']
    @property
    def min_num_layers(self):
        return self.random_architecture['min_num_layers']
    @property
    def max_num_layers(self):
        return self.random_architecture['max_num_layers']

    @property
    def dropout_enabled(self):
        return self.dropout['enabled']
    @property
    def dropout_prob(self):
        return self.dropout['p']

config = Config()