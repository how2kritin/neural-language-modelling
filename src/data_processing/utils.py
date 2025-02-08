import torch
from collections import Counter
from typing import List, Tuple


# util functions
def load_glove_embeddings(glove_file: str, vocab: dict, embedding_dim: int, device: str = 'cpu') -> torch.Tensor:
    embeddings = torch.zeros(len(vocab), embedding_dim)
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
            if word in vocab:
                embeddings[vocab[word]] = vector

    return embeddings.to(device)


def build_vocab(tokens: List[str]) -> Tuple[dict, dict]:
    word_counts = Counter(tokens)
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

    for word, _ in word_counts.most_common():
        if word not in vocab:
            vocab[word] = len(vocab)

    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word
