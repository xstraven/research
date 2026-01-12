import sys
from pathlib import Path

# Add src directory to path
# sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import polars as pl

from src.research.projects.connect4 import DATA_FOLDER


if __name__ == "__main__":
    print(DATA_FOLDER)
