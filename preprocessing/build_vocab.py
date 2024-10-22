import  pandas as pd
from torchtext.vocab import build_vocab_from_iterator
import nltk
import pickle

# Get the pd.series of comments
text_data = pd.read_csv("../data/raw/measuring-hate-speech-train.csv")["text"]

# Split them into tokens
#nltk.download('punkt_tab')
tokens = []
for comment in text_data:
    tokens.append(nltk.tokenize.word_tokenize(comment))

# Construct and save torchtext.vocab.vocab object for later indexing
vocab = build_vocab_from_iterator(tokens, specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"]) # Handles unknown tokens
with open("vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)