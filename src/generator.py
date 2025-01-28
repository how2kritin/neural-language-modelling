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

