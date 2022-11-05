from dataclasses import dataclass


@dataclass()
class TrainingParams:
    model_name: str
    random_state: int
