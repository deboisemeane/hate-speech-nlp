import datasets

dataset = datasets.load_dataset('ucberkeley-dlab/measuring-hate-speech')   
dataset['train'].to_csv('/data/raw/measuring-hate-speech-train.csv')
