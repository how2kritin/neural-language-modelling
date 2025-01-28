from tokenizer import word_tokenizer
import torch
from collections import defaultdict

def build_vocab(tokenized_sentences: list[list[str]]) -> dict:
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

def load_glove_embeddings(glove_file: str, vocab: dict, embedding_dim: int = 300) -> torch.Tensor:
    """
    To load glove embeddings into a PyTorch tensor.
    :param glove_file:
    :param vocab:
    :param embedding_dim: 300 by default.
    :return:
    """
    embeddings = torch.zeros(len(vocab), embedding_dim)
    with open(glove_file, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in vocab:
                vector = torch.tensor([float(val) for val in values[1:]], dtype=torch.float32)
                embeddings[vocab[word]] = vector
    return embeddings

def create_word_embeddings(tokenized_sentences: list[list[str]], vocab: dict, embeddings: torch.Tensor, window_size: int = 5) -> list[torch.Tensor]:
    """
    Create word embeddings of window_size using the loaded GloVe embeddings
    :param tokenized_sentences:
    :param vocab:
    :param embeddings:
    :param window_size:
    :return:
    """
    unk_idx = vocab['<UNK>']
    word_embeddings = []
    for sentence in tokenized_sentences:
        sentence_embeddings = []
        for i in range(len(sentence) - window_size + 1):
            window = sentence[i:i + window_size]
            window_embedding = []
            for word in window:
                word_lower = word.lower()
                if word_lower in vocab:
                    window_embedding.append(embeddings[vocab[word_lower]])
                else:
                    window_embedding.append(embeddings[unk_idx])  # using the <UNK> embedding

            # concatenate the embeddings for the {window_size}-word window
            window_embedding = torch.cat(window_embedding, dim=0)
            sentence_embeddings.append(window_embedding)
        word_embeddings.append(torch.stack(sentence_embeddings))
    return word_embeddings

def main():
    inp_sentence = str(input("your text: "))
    tokenized_sentences = word_tokenizer(inp_sentence)
    print("Tokenized text: ", tokenized_sentences)

    vocab = build_vocab(tokenized_sentences)
    print("Vocabulary: ", vocab)

    glove_file = 'path/to/glove.840B.300d.txt'
    embedding_dim = 300
    embeddings = load_glove_embeddings(glove_file, vocab, embedding_dim)
    print("Loaded embeddings shape: ", embeddings.shape)

    word_embeddings = create_word_embeddings(tokenized_sentences, vocab, embeddings)
    print("5-word embeddings (as PyTorch tensors): ", word_embeddings)

if __name__ == "__main__":
    main()