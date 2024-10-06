import datasets
import pandas as pd
from pathlib import Path


df = pd.read_csv('data/raw/measuring-hate-speech-train.csv')
print(df['text'])
