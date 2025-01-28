import argparse

from tokenizer import word_tokenizer
import torch
from collections import defaultdict
from typing import Literal


class WordEmbeddings:
    def __init__(self, tokenized_sentences: list[list[str]], glove_file_path: str, embedding_dim: int = 300,
                 window_size: int = 0, device: Literal['cpu', 'cuda'] = 'cpu'):
        """
        :param tokenized_sentences:
        :param glove_file_path:
        :param embedding_dim: 300 by default.
        :param window_size: 0 by default (indicating no window size needed). Set this to a non-zero integer if you want to create fixed window_size embeddings.
        :param device: "cpu" by default. Even if "cuda" is selected, if it is not available, then "cpu" will be chosen.
        """
        self.tokenized_sentences = tokenized_sentences
        self.glove_file_path = glove_file_path
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.word_embeddings = None
        self.glove_embeddings = None
        self.vocab = None

    def obtain_word_embeddings(self) -> list[torch.Tensor]:
        self.vocab = self._build_vocab(self.tokenized_sentences)
        self.glove_embeddings = self._load_glove_embeddings(self.glove_file_path, self.vocab, self.embedding_dim)
        self.word_embeddings = self._create_word_embeddings(self.tokenized_sentences, self.vocab, self.glove_embeddings,
                                                            self.window_size)

        return self.word_embeddings

    def _build_vocab(self, tokenized_sentences: list[list[str]]) -> dict:
        """
        Builds a vocabulary for words in the corpus.
        :param tokenized_sentences:
        :return:
        """
        vocab = defaultdict(lambda: len(vocab))
        vocab['<UNK>'] = 0  # reserving index 0 for unknown words
        for sentence in tokenized_sentences:
            for word in sentence:
                _ = vocab[word.lower()]
        return dict(vocab)

    def _load_glove_embeddings(self, glove_file: str, vocab: dict, embedding_dim: int = 300) -> torch.Tensor:
        """
        To load glove embeddings into a PyTorch tensor.
        :param glove_file:
        :param vocab:
        :param embedding_dim: 300 by default.
        :return:
        """
        embeddings = torch.zeros(len(vocab), embedding_dim, device=self.device)
        with open(glove_file, 'r', encoding='utf-8') as f:
            for line in f:
                values = line.split()
                word = values[0]
                if word in vocab:
                    try:
                        vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32,
                                              device=self.device)
                        embeddings[vocab[word]] = vector
                    except ValueError:
                        continue
        return embeddings

    def _create_word_embeddings(self, tokenized_sentences: list[list[str]], vocab: dict, embeddings: torch.Tensor,
                                window_size: int = 0) -> list[torch.Tensor]:
        """
        Create word embeddings of window_size using the loaded GloVe embeddings
        :param tokenized_sentences:
        :param vocab:
        :param embeddings:
        :param window_size: 0 by default (indicating no window size needed). Set this to a non-zero integer if you want to create fixed window_size embeddings.
        :return:
        """
        unk_idx = vocab['<UNK>']
        word_embeddings = []

        for sentence in tokenized_sentences:
            sentence_embeddings = []

            if window_size == 0:
                for word in sentence:
                    word_lower = word.lower()
                    sentence_embeddings.append(embeddings[vocab.get(word_lower, unk_idx)])

            else:
                # pad with <UNK> for short sentences
                if len(sentence) < window_size:
                    sentence = ['<UNK>'] * (window_size - len(sentence)) + sentence

                for i in range(len(sentence) - window_size + 1):
                    window = sentence[i:i + window_size]
                    window_embedding = [embeddings[vocab.get(word.lower(), unk_idx)] for word in window]

                    window_embedding = torch.cat(window_embedding, dim=0)  # Shape: (window_size * embedding_dim,)
                    sentence_embeddings.append(window_embedding)

            if sentence_embeddings:
                word_embeddings.append(torch.stack(sentence_embeddings))

        return word_embeddings


def main(corpus_path: str = None, window_size: int = 0, save_embeddings: bool = False):
    """
    :Ref: https://nlp.stanford.edu/projects/glove/ for Common Crawl (840B tokens, 2.2M vocab, cased, 300d vectors, 2.03 GB download)
    :return:
    """

    glove_file = '../glove.840B.300d.txt'
    embedding_dim = 300
    if not corpus_path:
        inp_sentence = str(input("your text: "))
    else:
        try:
            with open(corpus_path, "r") as file:
                inp_sentence = file.read()
        except FileNotFoundError:
            raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")

    tokenized_sentences = word_tokenizer(inp_sentence)

    we = WordEmbeddings(tokenized_sentences=tokenized_sentences, glove_file_path=glove_file,
                        embedding_dim=embedding_dim, window_size=window_size, device='cuda')
    we.obtain_word_embeddings()

    if not corpus_path:
        print("Tokenized text: ", tokenized_sentences)
        print("Vocabulary: ", we.vocab)
        print("Loaded embeddings shape: ", we.glove_embeddings.shape)
        print(f"{window_size}-word embeddings (as PyTorch tensors): ", we.word_embeddings)

    if save_embeddings:
        vocab_output_file = f"vocab.pt"
        glove_embeddings_output_file = f"glove_embeddings.pt"
        word_embeddings_output_file = f"word_embeddings_window-size={window_size}.pt"
        torch.save(we.vocab, vocab_output_file)
        torch.save(we.glove_embeddings, glove_embeddings_output_file)
        torch.save(we.word_embeddings, word_embeddings_output_file)
        print(f"Vocabulary saved to {vocab_output_file}, glove embeddings saved to {glove_embeddings_output_file} and word embeddings saved to {word_embeddings_output_file}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'w', type=int, help="Specify a window size."
    )
    parser.add_argument(
        '-c', type=str, default=None,
        help="Path to the corpus file. If not provided, interactive input will be used."
    )
    parser.add_argument(
        '-s', action='store_true',
        help="Flag to save the generated embeddings to a file. If not provided, it will default to False."
    )
    args = parser.parse_args()
    main(args.c, args.w, args.s)
