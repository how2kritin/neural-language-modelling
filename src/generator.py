import argparse
import math
import os.path
from tqdm import tqdm
import torch
import random
import numpy as np
from src.data_processing.tokenizer import word_tokenizer
from src.data_processing.utils import load_glove_embeddings
from src.data_processing.dataloaders import obtain_rnn_dataloaders, obtain_ffnn_dataloaders
from src.models.RNN import RNNLM
from src.models.FFNN import FFNNLM


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_perplexity_report(model, vocab, data_loader, output_file):
    model.eval()
    total_loss = 0.0
    count_sents = 0
    idx2word = {idx: word for word, idx in vocab.items()}

    lines = []

    for inputs, targets in tqdm(data_loader, desc="Calculating perplexities"):
        inputs, targets = inputs.to(model.device), targets.to(model.device)
        outputs = model(inputs)

        loss = model.loss_fn(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss = loss.view(targets.size())

        for i in range(inputs.size(0)):
            mask = (targets[i] != vocab['<PAD>']).float()

            num_valid_tokens = mask.sum()
            if num_valid_tokens > 0:
                sentence_loss = (loss[i] * mask).sum() / num_valid_tokens
                perplexity = torch.exp(sentence_loss).item()

                if not math.isnan(perplexity) and not math.isinf(perplexity):
                    total_loss += sentence_loss
                    count_sents += 1

                words = [idx2word[idx.item()] for idx in inputs[i] if idx.item() != vocab['<PAD>']]
                lines.append(f"{' '.join(words)}    {perplexity:.4f}\n")
            else:
                words = [idx2word[idx.item()] for idx in inputs[i] if idx.item() != vocab['<PAD>']]
                lines.append(f"{' '.join(words)}    N/A\n")

    avg_loss = total_loss / count_sents if count_sents > 0 else float('nan')
    avg_perplexity = torch.exp(avg_loss)

    with open(output_file, 'w') as f:
        f.write(f"{avg_perplexity:.4f}\n")
        f.writelines(lines)

    return avg_perplexity


def main(N: int, lm_type: str, corpus_path: str, k: int, model_path: str, task: str,
         perplexity_report_name: str) -> None:
    if lm_type == 'f' and not N:
        raise ValueError("Need to specify the N value for corpus size!")

    set_seed(42)  # setting a seed for reproducibility

    glove_file = 'glove.6B.300d.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        with open(corpus_path, "r") as file:
            text = file.read()
            tokenized_sentences = word_tokenizer(text)
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")

    if lm_type == 'r' or lm_type == 'l':
        vocab, idx2word, train_loader, valid_loader, test_loader = obtain_rnn_dataloaders(
            tokenized_sentences=tokenized_sentences, n_test_sents=1000, batch_size=32, max_seq_length=128)
    elif lm_type == 'f':
        vocab, idx2word, train_loader, valid_loader, test_loader = obtain_ffnn_dataloaders(
            tokenized_sentences=tokenized_sentences, n_test_sents=1000, n=N, batch_size=32)
    else:
        raise ValueError("Invalid lm_type!")

    glove_embeds = load_glove_embeddings(glove_file=glove_file, vocab=vocab, embedding_dim=300, device=device)

    match lm_type:
        case 'f':
            model_params = {'learning_rate': 5e-5, 'vocab': vocab, 'hidden_sizes': [N * 300],
                            'pretrained_embeds': glove_embeds, 'context_size': N - 1, 'dropout_rate': 0.3,
                            'n_epochs': 100, 'patience': 5, 'device': device}
            model = FFNNLM(**model_params)
        case 'r':
            model_params = {'learning_rate': 5e-5, 'vocab': vocab, 'hidden_size': 512, 'n_layers': 3,
                            'pretrained_embeds': glove_embeds, 'dropout_rate': 0.3, 'n_epochs': 100, 'patience': 5,
                            'device': device}
            model = RNNLM(**model_params, rnn_type='rnn')
        case 'l':
            model_params = {'learning_rate': 1e-4, 'vocab': vocab, 'hidden_size': 256, 'n_layers': 2,
                            'pretrained_embeds': glove_embeds, 'dropout_rate': 0.3, 'n_epochs': 100, 'patience': 5,
                            'device': device}
            model = RNNLM(**model_params, rnn_type='lstm')
        case _:
            raise ValueError("Please choose a valid model! Acceptable inputs: {f, r, l}")

    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Successfully loaded the pre-trained model!")
    else:
        print("Could not find a pre-trained model. Training model now...")
        model.train_model(train_loader, valid_loader)
        model.load_state_dict(torch.load('pretrained_models/best_model.pt'))

    if task == 'pe':
        if not perplexity_report_name: perplexity_report_name = "report"
        train_avg_ppl = generate_perplexity_report(model, vocab, train_loader,
                                                   f"perplexity_scores/{perplexity_report_name}_train-perplexities.txt")
        test_avg_ppl = generate_perplexity_report(model, vocab, test_loader,
                                                  f"perplexity_scores/{perplexity_report_name}_test-perplexities.txt")

        print(f"Training Set Average Perplexity: {train_avg_ppl:.4f}")
        print(f"Test Set Average Perplexity: {test_avg_ppl:.4f}")
    elif task == 'pr':
        if not k:  # by default get top 3 candidates if k isn't specified
            k = 3

        while True:
            inp_sentence = input("input sentence: ")
            words = word_tokenizer(inp_sentence)[0]

            indices = []
            for word in words:
                if word in vocab:
                    indices.append(vocab[word])
                else:
                    indices.append(vocab['<UNK>'])

            if lm_type == 'f':
                if len(indices) < N - 1:
                    padding_needed = N - 1 - len(indices)
                    indices = [vocab['<PAD>']] * padding_needed + indices
                else:
                    indices = indices[-(N - 1):]  # take last N-1 words if context is longer than N-1 words
            else:
                indices = [vocab['<BOS>']] + indices

            x = torch.tensor([indices], dtype=torch.long, device=device)

            indices_list, probs_list = model.predict_top_k(x, k)[0]  # take first batch
            for idx, prob in zip(indices_list, probs_list):
                word = idx2word[idx]
                print(f"{word} {prob:.3f}")
    else:
        raise ValueError("Invalid task!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, default=None,
                        help='N-gram size. Need to specify this only for FFNN.')
    parser.add_argument('-l', type=str, required=True, choices=['f', 'r', 'l'], help='LM type')
    parser.add_argument('-c', type=str, required=True, help='Path to the corpus')
    parser.add_argument('-k', type=int, required=False, default=None, help='Number of words to generate')
    parser.add_argument('-m', type=str, required=False, default=None, help='Path to the model (if it already exists)')
    parser.add_argument('-t', type=str, required=True, choices=['pe', 'pr'],
                        help='Task to perform (perplexity report generation or next word prediction)')
    parser.add_argument('-p', type=str, required=False, default=None,
                        help='Name of the perplexity report. Need to specify this only for task="pe".')
    args = parser.parse_args()
    main(args.n, args.l, args.c, args.k, args.m, args.t, args.p)
