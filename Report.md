## Name: Kritin Maddireddy

## Roll Number: 2022101071

---

---

> [!IMPORTANT]
> The Feed-Forward Neural Network LM has been trained on a learning rate of 5e-5, with a hidden size of `N * 300`, dropout rate of 0.3 for 100 epochs with patience of 5 epochs.
> The RNN LM has been trained on a learning rate of 5e-5, hidden size of 512 with 3 hidden layers, dropout rate of 0.3 for 100 epochs with patience of 5 epochs.
> The LSTM LM has been trained on a learning rate of 1e-4, hidden size of 256 with 2 hidden layers, dropout rate of 0.3 for 100 epochs with patience of 5 epochs.
> These hyperparameters have been chosen after tweaking each parameter to see what works best.

## Average Corpus Perplexity Scores:

| Model | N | Dataset             | Train | Test |
|-------|---|---------------------|-------|------|
| FFNN  | 3 | Pride and Prejudice | 38    | 162  |
| FFNN  | 3 | Ulysses             | 274   | 738  |
| FFNN  | 5 | Pride and Prejudice | 34    | 182  |
| FFNN  | 5 | Ulysses             | 173   | 684  |
| RNN   | - | Pride and Prejudice | 27    | 84   |
| RNN   | - | Ulysses             | 46    | 176  |
| LSTM  | - | Pride and Prejudice | 26    | 71   |
| LSTM  | - | Ulysses             | 51    | 174  |

## Results from the previous assignment for comparison:

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

