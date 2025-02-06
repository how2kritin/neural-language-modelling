import torch
from collections import Counter
from typing import List, Tuple
import random
from torch.utils.data import DataLoader, Dataset


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
    vocab = {word: idx for idx, (word, _) in enumerate(word_counts.items())}
    vocab['<UNK>'] = len(vocab)
    vocab['<PAD>'] = len(vocab)
    idx2word = {idx: word for word, idx in vocab.items()}
    return vocab, idx2word


# datasets
class RNNDataset(Dataset):
    def __init__(self, input_sentences, target_sentences, vocab: dict, seq_length: int = 20):
        super(RNNDataset, self).__init__()
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


class FFNNDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: dict, context_size: int):
        super(FFNNDataset, self).__init__()
        self.vocab = vocab
        self.context_size = context_size
        self.contexts = []
        self.targets = []

        for sentence in sentences:
            padded_sentence = ['<PAD>'] * (context_size - 1) + sentence
            word_ids = [vocab.get(w, vocab['<UNK>']) for w in padded_sentence]

            for i in range(len(word_ids) - context_size):
                context = word_ids[i:i + context_size]
                target = word_ids[i + context_size]

                if target != vocab['<PAD>']:
                    self.contexts.append(context)
                    self.targets.append(target)

    def __len__(self):
        return len(self.contexts)

    def __getitem__(self, idx):
        return torch.tensor(self.contexts[idx]), torch.tensor(self.targets[idx])


# helper functions to get datasets
def obtain_rnn_datasets(tokenized_sentences: list[list[str]], n_test_sents: int,
                        max_seq_len: int, batch_size: int) -> Tuple[dict, dict, DataLoader, DataLoader, DataLoader]:
    """
    :param tokenized_sentences:
    :param n_test_sents: Number of test sentences to consider.
    :param max_seq_len:
    :param batch_size:
    :return: vocabulary (dict), idx2word (dict), train, validation and test dataloaders.
    """
    random.shuffle(tokenized_sentences)

    total_size = len(tokenized_sentences)
    test_size = min(n_test_sents, total_size)

    remaining_data = total_size - test_size
    train_size = int(0.9 * remaining_data)
    val_size = remaining_data - train_size

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

    # split the sentences into train, test and validation
    input_train_sentences = input_tokenized_corpus[:train_size]
    input_valid_sentences = input_tokenized_corpus[train_size: train_size + val_size]
    input_test_sentences = input_tokenized_corpus[train_size + val_size:]

    target_train_sentences = target_tokenized_corpus[:train_size]
    target_valid_sentences = target_tokenized_corpus[train_size: train_size + val_size]
    target_test_sentences = target_tokenized_corpus[train_size + val_size:]

    print(f"num total_tokenized_{max_seq_len}_length_sentences: ", len(input_tokenized_corpus))
    print(f"num train_tokenized_{max_seq_len}_length_sentences: ", len(input_train_sentences))
    print(f"num valid_tokenized_{max_seq_len}_length_sentences: ", len(input_valid_sentences))
    print(f"num test_tokenized_{max_seq_len}_length_sentences: ", len(input_test_sentences))

    train_tokenized_corpus = tokenized_sentences[:train_size]
    tokens = []
    for sentence in train_tokenized_corpus:  # only include train vocab in the vocabulary
        tokens.extend(sentence)

    vocab, idx2word = build_vocab(tokens)
    print("Vocabulary created from training dataset, which has been randomly sampled from the corpus.")

    train_dataset = RNNDataset(input_train_sentences, target_train_sentences, vocab, seq_length=max_seq_len)
    valid_dataset = RNNDataset(input_valid_sentences, target_valid_sentences, vocab, seq_length=max_seq_len)
    test_dataset = RNNDataset(input_test_sentences, target_test_sentences, vocab, seq_length=max_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data Loaded Successfully")

    return vocab, idx2word, train_loader, valid_loader, test_loader


def obtain_ffnn_datasets(tokenized_sentences: List[List[str]], n_test_sents: int, context_size: int, batch_size: int):
    """
    :param tokenized_sentences: List of tokenized sentences
    :param n_test_sents: Number of test sentences
    :param context_size: Size of the context window
    :param batch_size: Batch size for DataLoader
    :return: vocabulary (dict), idx2word (dict), train, validation and test dataloaders.
    """
    random.shuffle(tokenized_sentences)

    total_size = len(tokenized_sentences)
    test_size = min(n_test_sents, total_size)

    remaining_data = total_size - test_size
    train_size = int(0.9 * remaining_data)
    val_size = remaining_data - train_size

    # split sentences into train, validation, and test sets
    train_sentences = tokenized_sentences[:train_size]
    valid_sentences = tokenized_sentences[train_size:train_size + val_size]
    test_sentences = tokenized_sentences[train_size + val_size:]

    print(f"Number of training sentences: {len(train_sentences)}")
    print(f"Number of validation sentences: {len(valid_sentences)}")
    print(f"Number of test sentences: {len(test_sentences)}")

    # build vocabulary only from training data
    tokens = []
    for sentence in train_sentences:
        tokens.extend(sentence)

    vocab, idx2word = build_vocab(tokens)
    print("Vocabulary created from training dataset, which has been randomly sampled from the corpus.")

    train_dataset = FFNNDataset(train_sentences, vocab, context_size)
    valid_dataset = FFNNDataset(valid_sentences, vocab, context_size)
    test_dataset = FFNNDataset(test_sentences, vocab, context_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data Loaded Successfully")
    print(f"Samples per epoch: {len(train_dataset)}")
    print(f"Context size: {context_size}")

    return vocab, idx2word, train_loader, valid_loader, test_loader
