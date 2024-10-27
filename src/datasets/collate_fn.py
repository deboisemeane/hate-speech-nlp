import torch
import pickle

def collate_fn(batch):
    """Function to collate batches into rectangular arrays of indexed words, using the padding token of the vocabulary to make
    all comments the same length.
    
    Args:
        batch (list): list of dictionaries("features", "score") returned by dataset.__getitem__
    """
    # Find the index which indicates padding
    with open("../../vocab.pkl") as f:
        vocab = pickle.load(f)
    padding_value = vocab["<pad>"]
    
    features_list = []
    score_list = []
    comment_lengths = [len(item["features"]) for item in batch]
    max_length = max(comment_lengths)
    
    # Pre-padding is more appropriate for RNNs
    for item in batch:
        
        features_list.append(example["features"])
        score_list.append(example["score"])
        