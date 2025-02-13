from typing import List, Tuple
import random
from torch.utils.data import DataLoader
from src.data_processing.utils import build_vocab
from src.data_processing.datasets import RNNDataset, FFNNDataset


# helper functions to get dataloaders
def _vocab_idxwordmap_train_valid_test_split(tokenized_sentences: list[list[str]], n_test_sents: int) -> Tuple[dict, dict, list[list[str]], list[list[str]], list[list[str]]]:
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

    return vocab, idx2word, train_sentences, valid_sentences, test_sentences


def obtain_rnn_dataloaders(tokenized_sentences: list[list[str]], n_test_sents: int, batch_size: int, max_seq_length: int = 128) -> Tuple[
    dict, dict, DataLoader, DataLoader, DataLoader]:
    """
    :param tokenized_sentences: List of tokenized sentences
    :param n_test_sents: Number of test sentences to consider
    :param batch_size: Batch size for DataLoader
    :param max_seq_length: Max length of each sentence (set this accordingly so that you don't run out of memory while training)
    :return: vocabulary (dict), idx2word (dict), train, validation and test dataloaders
    """
    vocab, idx2word, train_sentences, valid_sentences, test_sentences = _vocab_idxwordmap_train_valid_test_split(tokenized_sentences, n_test_sents)

    train_dataset = RNNDataset(train_sentences, vocab, max_seq_length)
    valid_dataset = RNNDataset(valid_sentences, vocab, max_seq_length)
    test_dataset = RNNDataset(test_sentences, vocab, max_seq_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=train_dataset.collate_batch)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                              collate_fn=valid_dataset.collate_batch)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=test_dataset.collate_batch)

    print("Data Loaded Successfully")
    return vocab, idx2word, train_loader, valid_loader, test_loader


def obtain_ffnn_dataloaders(tokenized_sentences: List[List[str]], n_test_sents: int, n: int, batch_size: int) -> Tuple[
    dict, dict, DataLoader, DataLoader, DataLoader]:
    """
    :param tokenized_sentences: List of tokenized sentences
    :param n_test_sents: Number of test sentences
    :param n: Size of the context window is 1 less than the 'n'-gram value (as for a 5-gram, 4 words are part of the context and 1 is target)
    :param batch_size: Batch size for DataLoader
    :return: vocabulary (dict), idx2word (dict), train, validation and test dataloaders.
    """
    vocab, idx2word, train_sentences, valid_sentences, test_sentences = _vocab_idxwordmap_train_valid_test_split(tokenized_sentences, n_test_sents)

    train_dataset = FFNNDataset(train_sentences, vocab, n - 1)
    valid_dataset = FFNNDataset(valid_sentences, vocab, n - 1)
    test_dataset = FFNNDataset(test_sentences, vocab, n - 1)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    print("Data Loaded Successfully")
    print(f"Context size (size of input window, i.e., 1 less than n-gram size): {n - 1}")

    return vocab, idx2word, train_loader, valid_loader, test_loader
