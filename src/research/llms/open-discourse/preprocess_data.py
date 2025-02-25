import re
from pathlib import Path
import pandas as pd

# Load dataset and do some light preprocessing

PATH_TO_DATA = Path("~/projects/open-discourse/python/data/03_final")
SPEECHES = "speech_content.pkl"
speeches = pd.read_pickle(PATH_TO_DATA / SPEECHES)
speeches.head()
speeches20 = speeches.loc[speeches.electoral_term == 20, :].copy()
## replace "({2})" with empty string
replace_regex = r"\(\{\d+\}\)"
speeches20.speech_content = speeches20.speech_content.apply(
    lambda x: re.sub(replace_regex, "", x)
)

## use regex to remove multiple spaces and newlines
speeches20.speech_content = speeches20.speech_content.apply(
    lambda x: re.sub(r"\s+", " ", x)
)
speeches20 = speeches20.loc[
    ~speeches20.position_long.isin(
        ["Vizepräsidentin", "Vizepräsident", "Präsidentin", "Präsident"]
    )
]
print(speeches20.iloc[0].speech_content)
speeches20.to_pickle("~/projects/research/data/od_speeches20.pkl")
