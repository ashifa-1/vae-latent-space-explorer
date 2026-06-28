import pandas as pd
from pathlib import Path


def save_training_log(logs, output_file="results/training_log.csv"):
    Path("results").mkdir(exist_ok=True)

    df = pd.DataFrame(logs)

    df.to_csv(output_file, index=False)