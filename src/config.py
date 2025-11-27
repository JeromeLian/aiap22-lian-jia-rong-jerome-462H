
from dataclasses import dataclass


@dataclass
class Config:
    db_path: str = "data/phishing.db"
    test_size: float = 0.2
    random_state: int = 42
    results_dir: str = "outputs"  # you can structure as you like (models/, plots/, etc.)


CONFIG = Config()
