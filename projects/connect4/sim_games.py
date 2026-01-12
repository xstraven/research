import sys
from pathlib import Path

# Add projects directory to path when run as a script
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent.parent))

import polars as pl

try:
    from . import DATA_FOLDER
except ImportError:
    # When run as a script, use absolute import
    from connect4 import DATA_FOLDER


if __name__ == "__main__":
    print(DATA_FOLDER)
