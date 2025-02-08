## Name: Kritin Maddireddy

## Roll Number: 2022101071

---

---

> [!IMPORTANT]  
> The Feed-Forward Neural Network LM has been trained on a learning rate of 5e-5, with a hidden size of `N * 300`,
> dropout rate of 0.3 for 100 epochs with patience of 5 epochs.
> The RNN LM has been trained on a learning rate of 5e-5, hidden size of 512 with 3 hidden layers, dropout rate of 0.3
> for 100 epochs with patience of 5 epochs.
> The LSTM LM has been trained on a learning rate of 1e-4, hidden size of 256 with 2 hidden layers, dropout rate of 0.3
> for 100 epochs with patience of 5 epochs.
> These hyperparameters have been chosen after tweaking each parameter to see what works best.

---

## Perplexity scores

### Average Corpus Perplexity Scores:

| Model | N | Dataset             | Train | Test |
|-------|---|---------------------|-------|------|
| FFNN  | 3 | Pride and Prejudice | 33    | 128  |
| FFNN  | 3 | Ulysses             | 169   | 405  |
| FFNN  | 5 | Pride and Prejudice | 41    | 144  |
| FFNN  | 5 | Ulysses             | 213   | 397  |
| RNN   | - | Pride and Prejudice | 27    | 84   |
| RNN   | - | Ulysses             | 46    | 176  |
| LSTM  | - | Pride and Prejudice | 26    | 71   |
| LSTM  | - | Ulysses             | 51    | 174  |

### Results from the previous assignment for comparison:

| Model | N | Dataset           | Smoothing            | Train | Test  |
|-------|---|-------------------|----------------------|-------|-------|
| LM1   | 1 | Pride & Prejudice | Laplace              | 414   | 416   |
| LM1   | 3 | Pride & Prejudice | Laplace              | 1781  | 3033  |
| LM1   | 5 | Pride & Prejudice | Laplace              | 2475  | 4710  |
| LM2   | 1 | Pride & Prejudice | Good-Turing          | 598   | 689   |
| LM2   | 3 | Pride & Prejudice | Good-Turing          | 20    | 70    |
| LM2   | 5 | Pride & Prejudice | Good-Turing          | 13    | 50    |
| LM3   | 1 | Pride & Prejudice | Linear Interpolation | 413   | 459   |
| LM3   | 3 | Pride & Prejudice | Linear Interpolation | 11    | 186   |
| LM3   | 5 | Pride & Prejudice | Linear Interpolation | 7     | 189   |
| LM4   | 1 | Ulysses           | Laplace              | 821   | 1059  |
| LM4   | 3 | Ulysses           | Laplace              | 8963  | 15575 |
| LM4   | 5 | Ulysses           | Laplace              | 11536 | 21430 |
| LM5   | 1 | Ulysses           | Good-Turing          | 2200  | 3676  |
| LM5   | 3 | Ulysses           | Good-Turing          | 129   | 235   |
| LM5   | 5 | Ulysses           | Good-Turing          | 176   | 298   |
| LM6   | 1 | Ulysses           | Linear Interpolation | 832   | 1299  |
| LM6   | 3 | Ulysses           | Linear Interpolation | 16    | 955   |
| LM6   | 5 | Ulysses           | Linear Interpolation | 12    | 962   |

---

## Analysis

### Performance Ranking of Models (Best to Worst)

1. LSTM models
2. RNN models
3. N-gram models with Good-Turing Smoothing
4. FFNN-based models
5. N-gram models with Linear Interpolation
6. N-gram models with Laplace Smoothing

### Comparative Analysis

#### LSTM models

These perform the best overall, as they can capture AND remember longer-range dependencies better than other neural
network models. For both the datasets, these models perform better. This is due to their architecture employing the use
of "gates" that control the flow of info into and out of an LSTM memory cell, thereby improving their ability to
remember. However, this also means that they train slower than typical RNN models, as they have more states to manage.

In fact, these models have the smallest gap between train and test perplexities, showing that they generalize better.

#### RNN models

These are slightly worse than LSTM-based models, primarily because they tend to forget info as new info comes in.
However, they still are much better than Feed-Forward Neural Network models, as they can observe relatively long-range
dependencies as they do not need to depend on a context window.

#### FFNN-based models

While these models are, on average, better than most n-gram models, they seem to perform worse than N-gram models with
Good-Turing Smoothing, going purely based off of the perplexity scores. This could be due to them suffering from neural
network optimization challenges. However, I believe that with the right set of hyperparameters, these could perform
at least on par with those N-gram models with Good-Turing smoothing. They still are, however, worse than RNN-based
models (either RNN/LSTM layers).

Good-Turing seems to work better here, due to an efficient probability estimation method (which is pretty comprehensive)
for rare events, and honestly, it doesn't require much hyperparameter tuning; they work well right out of the box.

### Observations

#### Which model performs better for longer sentences? Why?

For longer sentences (i.e., sentences from the Ulysses dataset, which are longer and more complex on average),
LSTM models hit other models out of the park, due to their ability to "remember" and observe long-range dependencies.

RNNs are decent as well. However, the difference in perplexity scores between them isn't large, primarily because the "
long-term" dependencies in Ulysses aren't as long, rather, local
context seems to be more important for this dataset. The complexity arises more from vocabulary than from long-range
dependencies. (hence, on a pre-trained word embedding method, a lot of words in this dataset are tagged as `<UNK>`.
Perhaps, in a much larger GloVe embedding size, some of these words may exist.)

However, judging by the perplexity scores, it seems as though the Pride and Prejudice dataset has more long-range
dependencies, as compared to the Ulysses dataset. Here, we see a clear winner; LSTM models can "remember" context
better, and thus they perform better than their RNN counterparts.

#### How does the choice of n-gram size affect the performance of the FFNN model?

For FFNN models, a larger N implies that there's more input features, and more parameters to learn. However, there is a
higher risk of overfitting. More complex text, such as the Ulysses dataset, **slightly** (again, not by much) benefits
from a larger context, but simpler text, such as the Pride and Prejudice dataset, performs better with smaller context.
Here, as Pride and Prejudice dataset has a more standard language with regular patterns, larger N tends to hurt
performance and simpler patterns are sufficient (and larger context windows seem to introduce the model to spurious
patterns). On the contrary, for a language with non-standard patterns and complex vocabulary, larger N helps slightly
(there's a benefit from extra context).

Hence, for standard text, prefer a smaller context window (like N = 3) and for more complex text, prefer a larger
context window (like N = 5). 

---
