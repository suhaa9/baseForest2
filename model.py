from dataclasses import dataclass
from typing import Sequence, Tuple

from sklearn.neural_network import MLPClassifier


@dataclass
class MLPConfig:
    hidden_layer_sizes: Sequence[int] = (512, 256)
    activation: str = 'relu'
    solver: str = 'adam'
    learning_rate_init: float = 1e-3
    batch_size: int = 64
    random_state: int | None = 42


def build_classifier(input_dim: int, num_classes: int, *, config: MLPConfig | None = None) -> MLPClassifier:
    cfg = config or MLPConfig()
    return MLPClassifier(
        hidden_layer_sizes=tuple(cfg.hidden_layer_sizes),
        activation=cfg.activation,
        solver=cfg.solver,
        learning_rate_init=cfg.learning_rate_init,
        batch_size=cfg.batch_size,
        max_iter=1,
        warm_start=True,
        random_state=cfg.random_state,
        shuffle=True,
    )
