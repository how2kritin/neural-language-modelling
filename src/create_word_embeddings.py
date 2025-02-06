import torch
from collections import Counter
from typing import List, Tuple
import random
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence


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


# datasets
class RNNDataset(Dataset):
    """
    Reference on choosing labels and targets for time series data like RNNs: https://stackoverflow.com/questions/47008349/how-to-choose-label-target-for-rnn-models
    """

    def __init__(self, sequences: list[list[str]], vocab: dict):
        super(RNNDataset, self).__init__()
        self.sequences = []
        self.vocab = vocab

        for sequence in sequences:
            sequence_with_tokens = ['<BOS>'] + sequence + ['<EOS>']

            indices = [self.vocab.get(x, self.vocab['<UNK>']) for x in sequence_with_tokens]

            input_seq = indices[:-1]
            target_seq = indices[1:]

            self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

    @staticmethod
    def collate_batch(batch):
        inputs, targets = zip(*batch)

        inputs_padded = pad_sequence([torch.tensor(x) for x in inputs],
                                     batch_first=True,
                                     padding_value=0)
        targets_padded = pad_sequence([torch.tensor(y) for y in targets],
                                      batch_first=True,
                                      padding_value=0)

        return inputs_padded, targets_padded


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
                        batch_size: int) -> Tuple[dict, dict, DataLoader, DataLoader, DataLoader]:
    """
    :param tokenized_sentences: List of tokenized sentences
    :param n_test_sents: Number of test sentences to consider
    :param batch_size: Batch size for DataLoader
    :return: vocabulary (dict), idx2word (dict), train, validation and test dataloaders
    """
    random.shuffle(tokenized_sentences)

    total_size = len(tokenized_sentences)
    test_size = min(n_test_sents, total_size)

    remaining_data = total_size - test_size
    train_size = int(0.9 * remaining_data)
    val_size = remaining_data - train_size

    train_sentences = tokenized_sentences[:train_size]
    valid_sentences = tokenized_sentences[train_size:train_size + val_size]
    test_sentences = tokenized_sentences[train_size + val_size:]

    print(f"Number of training sentences: {len(train_sentences)}")
    print(f"Number of validation sentences: {len(valid_sentences)}")
    print(f"Number of test sentences: {len(test_sentences)}")

    # build vocabulary from training data only
    tokens = []
    for sentence in train_sentences:
        tokens.extend(sentence)

    vocab, idx2word = build_vocab(tokens)
    print("Vocabulary created from training dataset, which has been randomly sampled from the corpus.")

    train_dataset = RNNDataset(train_sentences, vocab)
    valid_dataset = RNNDataset(valid_sentences, vocab)
    test_dataset = RNNDataset(test_sentences, vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=train_dataset.collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=valid_dataset.collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             collate_fn=test_dataset.collate_batch)

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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data Loaded Successfully")
    print(f"Context size: {context_size}")

    return vocab, idx2word, train_loader, valid_loader, test_loader
