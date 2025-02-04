# import argparse
# from tokenizer import word_tokenizer
# import random
# import re
# from LSTM import LSTMLM
#
# random_state = 42  # for train-test split reproducibility while obtaining perplexity
# random.seed(random_state)
#
#
# # function to detokenize a list of tokens
# def detokenize(tokens):
#     sentence = " ".join(tokens)
#     # fixing spaces before punctuation
#     sentence = re.sub(r"\s+([.,!?;:\"\')])", r"\1", sentence)
#     # fixing spaces after opening quotes/brackets
#     sentence = re.sub(r"([\"'(\[{])\s+", r"\1", sentence)
#     return sentence
#
#
# def calculate_and_save_perplexities(sentences: list[list[str]], ngm: NGramModel | LinearInterpolationOfNGramModels,
#                                     output_file: str) -> None:
#     total_perplexity = 0
#     num_sentences = len(sentences)
#
#     results = []
#     for sentence in sentences:
#         sentence_perplexity = ngm.perplexity(sentence)
#         total_perplexity += sentence_perplexity
#         results.append((detokenize(sentence), sentence_perplexity))
#
#     avg_perplexity = total_perplexity / num_sentences
#
#     with open(output_file, 'w', encoding='utf-8') as f:
#         f.write(f"{avg_perplexity}\n")
#         for sentence_text, perplexity in results:
#             f.write(f"{sentence_text}\t{perplexity}\n")
#
#
# def main(N: int, lm_type: str, vocab_path: str, glove_embeddings_path: str, word_embeddings_path: str, k: int, gen_type: str) -> None:
#     hyperparameters = {'learning_rate': 1e-3, }
#     match lm_type:
#         case 'f':
#             pass
#         case 'r':
#             pass
#         case 'l':
#             lm = LSTM(learning_rate=)
#         case _:
#             raise ValueError("Please choose a valid model! Acceptable inputs: {f, r, l}")
#
#     try:
#         with open(corpus_path, "r") as file:
#             text = file.read()
#         tokenized_sentences = word_tokenizer(text)
#     except FileNotFoundError:
#         raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")
#
#     if smoothing_type == 'linear_interpolation':
#         ngm = LinearInterpolationOfNGramModels(N)
#     else:
#         ngm = NGramModel(N=N, smoothing_type=smoothing_type)
#
#     ngm.train(tokenized_sentences)
#     input_sentence = str(input('input sentence: '))
#
#     if gen_type == 's':
#         generated_list_of_words = ngm.generate_sentence_next_n_words(
#             tokenized_sentence=word_tokenizer(input_sentence)[0], n=k)
#         print(detokenize(generated_list_of_words))
#     elif gen_type == 'w':
#         predicted_words_dict = ngm.predict_next_word(tokenized_sentence=word_tokenizer(input_sentence)[0],
#                                                      n_candidates_for_next_word=k)
#         if len(predicted_words_dict) == 0:
#             print("Could not predict any possible candidates for the next word.")
#             return
#
#         print("output:")
#         for key, val in predicted_words_dict.items():
#             print(key, val)\
import argparse
import math
import os.path
from tqdm import tqdm
from tokenizer import word_tokenizer
from create_word_embeddings import obtain_train_test_val_datasets, load_glove_embeddings
from LSTM import LSTMLM
import torch
import random
import numpy as np


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def generate_perplexity_report(model, vocab, data_loader, output_file):
    model.eval()
    total_perplexity = 0.0
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
                    total_perplexity += perplexity
                    count_sents += 1

                words = [idx2word[idx.item()] for idx in inputs[i] if idx.item() != vocab['<PAD>']]
                lines.append(f"{' '.join(words)}    {perplexity:.4f}\n")
            else:
                words = [idx2word[idx.item()] for idx in inputs[i] if idx.item() != vocab['<PAD>']]
                lines.append(f"{' '.join(words)}    N/A\n")

    avg_perplexity = total_perplexity / count_sents if count_sents > 0 else float('nan')

    with open(output_file, 'w') as f:
        f.write(f"{avg_perplexity:.4f}\n")
        f.writelines(lines)

    return avg_perplexity


def main(N: int, lm_type: str, corpus_path: str, k: int, model_path: str, task: str, perplexity_report_name: str) -> None:
    set_seed(42)

    glove_file = '../glove.6B.300d.txt'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        with open(corpus_path, "r") as file:
            text = file.read()
            tokenized_sentences = word_tokenizer(text)
    except FileNotFoundError:
        raise FileNotFoundError("Unable to find a file at that path to use as the corpus!")

    vocab, idx2word, train_loader, valid_loader, test_loader = obtain_train_test_val_datasets(
        tokenized_sentences=tokenized_sentences, n_test_sents=1000, max_seq_len=50, batch_size=32)
    glove_embeds = load_glove_embeddings(glove_file=glove_file, vocab=vocab, embedding_dim=300, device=device)

    model_params = {'learning_rate': 5e-5, 'vocab': vocab, 'hidden_size': 512, 'n_layers': 3, 'embedding_dim': 300,
                    'pretrained_embeds': glove_embeds, 'dropout_rate': 0.3, 'n_epochs': 100, 'patience': 5,
                    'device': device}
    model = LSTMLM(**model_params)
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
    else:
        print("Could not find a pre-trained model. Training model now...")
        model.train_model(train_loader, valid_loader)
        model.load_state_dict(torch.load('best_model.pt'))

    if task == 'pe':
        if not perplexity_report_name: perplexity_report_name = "report"
        train_avg_ppl = generate_perplexity_report(model, vocab, train_loader,
                                                   f"../perplexity_scores/{perplexity_report_name}_train-perplexities.txt")
        test_avg_ppl = generate_perplexity_report(model, vocab, test_loader,
                                                  f"../perplexity_scores/{perplexity_report_name}_test-perplexities.txt")

        print(f"Training Set Average Perplexity: {train_avg_ppl:.4f}")
        print(f"Test Set Average Perplexity: {test_avg_ppl:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', type=int, required=False, default=None, help='N-gram size. Need to specify this only for FFNN.')
    parser.add_argument('-l', type=str, required=True, choices=['f', 'r', 'l'], help='LM type')
    parser.add_argument('-c', type=str, required=True, help='Path to the corpus')
    parser.add_argument('-k', type=int, required=False, default=None, help='Number of words to generate')
    parser.add_argument('-m', type=str, required=False, default=None, help='Path to the model (if it already exists)')
    parser.add_argument('-t', type=str, required=True, choices=['pe', 'pr'], help='Task to perform (perplexity report generation or next word prediction)')
    parser.add_argument('-p', type=str, required=False, default=None, help='Name of the perplexity report. Need to specify this only for task="pe".')
    args = parser.parse_args()
    main(args.n, args.l, args.c, args.k, args.m, args.t, args.p)

