# Neural Language Modelling

Neural Network models, such as Feed-Forward Neural Network, Recurrent Neural Network and Long-Short Term Memory neural
network used for language modelling. Implemented in Python using PyTorch. This corresponds to Assignment-2 of the
Introduction to Natural Language Processing course at IIIT Hyderabad, taken in the Spring'25 semester.

---

# Pre-requisites

1. `python 3.12`
2. A python package manager such as `pip` or `conda`.
3. [pytorch](https://pytorch.org/get-started/locally/)
4. (OPTIONAL) `virtualenv` to create a virtual environment.
5. All the python libraries mentioned in `requirements.txt`.
6. Pre-trained [GloVe](https://nlp.stanford.edu/projects/glove/) word embeddings (assumed to be `glove.6B.300d`, change
   it if required in the `main()` function in `src/generator.py`).

---

# Tokenization

Using regex to tokenize corpus into list of sentences, where each sentence is a list of
tokens. Additionally, processing URLs, mentions, ages, hashtags and time data.

---

# Generation

## Instructions to run

### Usage

```bash
python3 -m src.generator -l <lm_type> -c <corpus_path> -t <task> [-n <N>] [-k <num_words>] [-m <model_path>] [-p <report_name>]
```

### Arguments

- `-l <lm_type>` : Specifies the language model type. Must be one of:
    - `'f'` : Feed Forward Neural Network LM (FFNN LM)
    - `'r'` : Recurrent Neural Network LM (RNN LM)
    - `'l'` : Long Short-Term Memory LM (LSTM LM)
- `-c <corpus_path>` : Path to the corpus file.
- `-t <task>` : Task to perform. Must be one of:
    - `'pr'` : Next-word prediction (requires a sentence input).
    - `'pe'` : Generate a perplexity report.

### Optional Arguments

- `-n <N>` : N-gram size (only required for FFNN models).
- `-k <num_words>` : Number of candidates to predict for the next word, given a context.
- `-m <model_path>` : Path to a pre-existing model (if available).
- `-p <report_name>` : Name of the perplexity report (only required for `pe` task).

### Example Usage

```bash
python3 -m src.generator -l f -c corpus/corpus.txt -t pr
```

```bash
python3 -m src.generator -l r -c corpus/corpus.txt -t pe -p report.txt
```

## How to obtain perplexity?

Run ```python3 language_model.py <N> <lm_type> <corpus_path> pe``` to automatically generate two files at
`./perplexity_scores/` directory; one for the train set and another for the test set. The number at the top of the file
is avg perplexity of the corpus, followed by `<sentence_i>   <perplexity>` lines.

To calculate perplexity, I am using the formula:

$$PP(w) = 2^{-\frac{1}{N}\sum {\log_2 P(w_i|w_{i-n+1}...w_{i-1}})}$$

instead of:

$$PP(w) = \root{N}\of{\frac{1}{\prod{P(w_i|w_{i-n+1}...w_{i-1})}}}$$

for the sake of numerical stability.

> [!NOTE]
> For neural network models, if using the cross-entropy loss function, then perplexity is simply $\exp(loss)$. This
> happens to be the case in my implementation of RNN and FFNN models.

---

# Analysis

Please refer to the [report](Report.md) for an analysis of these models.

---

