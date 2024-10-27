import torch
import pandas as pd
import pickle


class Dataset(torch.utils.data.Dataset):
    
    def __init__(self):
        self.df = pd.read_csv("data/processed/measuring-hate-speech-train.csv")
        with open("../../vocab.pkl", "rb") as f:
            self.vocab = pickle.load(f)
        
    
    def __getitem__(self, idx):
        features = self.df.iloc[idx]["text_indexed"]
        score = self.df.iloc[idx]["hate_speech_score"]
        features = torch.tensor(features, torch.float32)
        score = torch.tensor(score, torch.float32)
        return {"features": features, "score": score}

