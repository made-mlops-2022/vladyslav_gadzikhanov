from dataclasses import dataclass


@dataclass()
class SplittingParams:
    test_size: float = 0.2
    random_state: int = 42
