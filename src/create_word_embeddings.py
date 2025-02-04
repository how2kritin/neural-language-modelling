import torch
from collections import Counter
from typing import List, Tuple
import random
from torch.utils.data import DataLoader, Dataset
from tokenizer import word_tokenizer


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
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word


# dataset
class CustomDataset(Dataset):
    def __init__(self, input_sentences, target_sentences, vocab: dict, seq_length: int = 20):
        super(CustomDataset, self).__init__()
        self.input_sentences = input_sentences
        self.target_sentences = target_sentences
        self.seq_length = seq_length
        self.vocab = vocab

    def __len__(self):
        return len(self.input_sentences)

    def __getitem__(self, idx):
        input_idxs = [self.vocab.get(w, self.vocab['<UNK>']) for w in self.input_sentences[idx]]
        target_idxs = [self.vocab.get(w, self.vocab['<UNK>']) for w in self.target_sentences[idx]]

        return torch.tensor(input_idxs), torch.tensor(target_idxs)


def obtain_train_test_val_datasets(tokenized_sentences: list[list[str]], train_fraction: float, test_fraction: float,
                                   max_seq_len: int, batch_size: int):
    random.shuffle(tokenized_sentences)

    total_size = len(tokenized_sentences)
    train_size = int(train_fraction * total_size)
    test_size = int(test_fraction * total_size)
    val_size = total_size - (train_size + test_size)

    pad_token = "<PAD>"

    input_tokenized_corpus = []
    target_tokenized_corpus = []

    for sentence in tokenized_sentences:
        for i in range(0, len(sentence), max_seq_len - 1):
            input_seq = sentence[i: i + max_seq_len - 1]

            if len(input_seq) < max_seq_len:
                input_seq += [pad_token] * (max_seq_len - len(input_seq))

            target_seq = input_seq[1:] + [pad_token]

            input_tokenized_corpus.append(input_seq)
            target_tokenized_corpus.append(target_seq)

    print("Input Sequences: ", input_tokenized_corpus[:1])
    print("len: ", len(input_tokenized_corpus[0]))
    print("Target Sequences: ", target_tokenized_corpus[:1])
    print("len: ", len(target_tokenized_corpus[0]))

    # split the sentences into train, test and validation
    input_train_sentences = input_tokenized_corpus[:train_size]
    input_valid_sentences = input_tokenized_corpus[train_size: train_size + val_size]
    input_test_sentences = input_tokenized_corpus[train_size + val_size:]
    target_train_sentences = target_tokenized_corpus[:train_size]
    target_valid_sentences = target_tokenized_corpus[train_size: train_size + val_size]
    target_test_sentences = target_tokenized_corpus[train_size + val_size:]

    print("total_sentences: ", total_size)
    print("train_sentences: ", len(input_train_sentences))
    print("valid_sentences: ", len(input_valid_sentences))
    print("test_sentences: ", len(input_test_sentences))
    print("train_sentences: ", len(target_train_sentences))
    print("valid_sentences: ", len(target_valid_sentences))
    print("test_sentences: ", len(target_test_sentences))

    train_tokenized_corpus = tokenized_sentences[:train_size]
    tokens = []
    for sentence in train_tokenized_corpus:  # only include train vocab in the vocabulary
        tokens.extend(sentence)

    print("Text file preprocessed")
    vocab, idx2word = build_vocab(tokens)
    print("Vocabulary created for training dataset, which has been randomly sampled from the corpus.")

    generator = torch.Generator(device='cpu')

    train_dataset = CustomDataset(input_train_sentences, target_train_sentences, vocab, seq_length=max_seq_len)
    valid_dataset = CustomDataset(input_valid_sentences, target_valid_sentences, vocab, seq_length=max_seq_len)
    test_dataset = CustomDataset(input_test_sentences, target_test_sentences, vocab, seq_length=max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, generator=generator)
    print("Data Loaded Successfully")

    return vocab, idx2word, train_loader, valid_loader, test_loader