import torch
from typing import List
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


# datasets
import torch
from typing import List
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class RNNDataset(Dataset):
    """
    Reference on choosing labels and targets for time series data like RNNs: https://stackoverflow.com/questions/47008349/how-to-choose-label-target-for-rnn-models
    """

    def __init__(self, sequences: list[list[str]], vocab: dict, max_seq_length: int = 50):
        super(RNNDataset, self).__init__()
        self.sequences = []
        self.vocab = vocab

        for sequence in sequences:
            # adding BOS and EOS only once for the full sequence
            sequence_with_tokens = ['<BOS>'] + sequence + ['<EOS>']
            indices = [self.vocab.get(x, self.vocab['<UNK>']) for x in sequence_with_tokens]

            # break the full tokenized sequence into chunks of max_seq_length
            for i in range(0, len(indices), max_seq_length):
                chunk = indices[i:i + max_seq_length + 1]  # +1 to include target for last token
                if len(chunk) > 1:
                    input_seq = chunk[:-1]
                    target_seq = chunk[1:]
                    self.sequences.append((input_seq, target_seq))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        input_seq, target_seq = self.sequences[idx]
        return torch.tensor(input_seq), torch.tensor(target_seq)

    @staticmethod
    def collate_batch(batch):
        inputs, targets = zip(*batch)

        inputs_padded = pad_sequence([torch.tensor(x) for x in inputs], batch_first=True, padding_value=0)
        targets_padded = pad_sequence([torch.tensor(y) for y in targets], batch_first=True, padding_value=0)

        return inputs_padded, targets_padded


class FFNNDataset(Dataset):
    def __init__(self, sentences: List[List[str]], vocab: dict, context_size: int):
        super(FFNNDataset, self).__init__()
        self.vocab = vocab
        self.context_size = context_size
        self.contexts = []
        self.targets = []

        for sentence in sentences:
            sentence = ['<BOS>'] + sentence + ['<EOS>']
            # padding sentence to ensure minimum context size
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
