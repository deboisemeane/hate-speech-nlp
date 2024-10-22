import nltk
import torchtext
import pandas as pd
import pickle

df = pd.read_csv("data/raw/measuring-hate-speech-train.csv")
# Drop duplicate comments as for now we only care about aggregate hate_speech score
# Currently we don't care about different annotators
df = df.drop_duplicates(subset="comment_id")
print(df.shape)
# Load vocab object
with open("vocab.pkl", "rb") as f:
    vocab = pickle.load(f)

# Tokenize and Index
indexes = []
for comment in df["text"]:
    tokens = nltk.tokenize.word_tokenize(comment)
    indexes.append(vocab.forward(tokens))

# Insert 
df.insert(1, "text_indexed", indexes)
print(df.head())
df = df[["comment_id", "text_indexed", "text", "hate_speech_score"]]
df.to_csv("data/processed/measuring-hate-speech-train.csv", index=False)



