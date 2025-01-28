# Neural Language Modeling

# Resources

# Pre-requisites

1. `python 3.12`
2. A python package manager such as `pip` or `conda`.
3. [pytorch](https://pytorch.org/get-started/locally/)
4. (OPTIONAL) `virtualenv` to create a virtual environment.
5. All the python libraries mentioned in `requirements.txt`.

# Tokenization



---

# Smoothing and Interpolation

## Instructions to run

```python3 language_model.py <N> <lm_type> <corpus_path> <task>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear
Interpolation, 'n' for No Smoothing.  
`<N>` is the $N$-gram size, `<corpus_path>` is the path to the corpus and `<task>` is either 'pr' to get probability of
sentence (must provide a sentence as input when prompted), or `pe` to get the avg perplexity of the corpus and the
perplexity of 1000 train and 1000 test sentences.

### How to obtain perplexity?

Run ```python3 language_model.py <N> <lm_type> <corpus_path> pe``` to automatically generate two files at
`./perplexity_scores/` directory; one for the train set and another for the test set. The number at the top of the file
is avg perplexity of the corpus, followed by `<sentence_i>   <perplexity>` lines.

To calculate perplexity, I am using the formula:

$$PP(w) = 2^{-\frac{1}{N}\sum {\log_2 P(w_i|w_{i-n+1}...w_{i-1}})}$$

instead of:

$$PP(w) = \root{N}\of{\frac{1}{\prod{P(w_i|w_{i-n+1}...w_{i-1})}}}$$

for the sake of numerical stability.

## Different types of smoothing used, and their logic

### Laplace Smoothing

$$P(w|h) = \frac{c(w,h) + 1}{c(h) + V}$$

where $V$ is the total vocabulary size (assumed known).

Essentially, we pretend that we saw every word once more than we actually did. Hence, it is also called "Add-One"
smoothing.

### Good-Turing Smoothing

$$P_{GT}(w_1...w_n) = \frac{r^*}{N}$$

where:

$$r^* = \frac{(r + 1)S(N_{r + 1})}{S(N_r)}$$

and

$$N = \sum rN_r$$

Here, $S(\cdot)$ is the smoothed function. For small values of $r$, $S(N_r) = N_r$ is a reasonable assumption (no
smoothing is performed). N is the total number of objects observed, i.e., it is the total number of $n$-grams. However,
for larger values of $r$, values of $S(N_r)$ are read off the regression line given by
the logarithmic relationship

$$log(N_r) = a + blog(r)$$

where $N_r$ is the number of times $n$-grams of frequency $r$
have occurred.

However, this plot of $log(N_r)$ versus $log(r)$ is problematic because for large $r$, many $N_r$ will be zero. Instead,
we plot a revised quantity, $log(Z_r)$ versus $log(r)$, where $Z_r$ is defined as

$$Z_r = \frac{N_r}{\frac{1}{2}(t - q)}$$

and where $q$, $r$ and $t$ are three consecutive subscripts with non-zero counts $N_q$, $N_r$, $N_t$. For the special
case where $r$ is 1, we take $q = 0$. In the opposite special case, when $r = r_{last}$ is the index of the _last_
non-zero count, replace the divisor $\frac{1}{2}(t-q)$ with $r_{last}-q$.

**For unseen events:**

$$P_{GT}(w_1...w_n) = \frac{N_1}{N}$$

Here, we are using the _Turing_ estimate for small $r$ values, and the _Good-Turing_ estimate for large $r$ values.
Since we are combining two different estimates of probabilities, we do not expect
them to add to one. In this condition, our estimates are called _unnormalized_. We make sure that the
probability estimates add to one by dividing by the total of the _unnormalized_ estimates. This is called
_renormalization_.

$$P_{SGT} = (1 - \frac{N_1}{N}) \frac{P^{unnorm}_r}{\sum P^{unnorm}_r} \hspace{5mm}r \geq 1$$

This renormalized estimate is the _Simple Good-Turing_ (SGT) smoothing estimate. This is what we will be using here.

> [!NOTE]
> As can be seen in the renormalized probability above, the probability mass $\frac{N_1}{N}$ is reserved for
unseen events, and hence, when predicting the next word, the probabilities of _ALL_ the predicted words will sum up
to $1 - \frac{N_1}{N}$.

### Linear Interpolation

When performing this for a trigram, we estimate the trigram probabilities as follows:

$$P(t_3|t_1, t_2) = \lambda_1\hat{P}(t_3) + \lambda_2\hat{P}(t_3|t_2) + \lambda_3\hat{P}(t_3|t_1, t_2)$$

where $\hat{P}$ are the maximum likelihood estimates of the probabilities and $\lambda_1 + \lambda_2 + \lambda_3 = 1$
so $P$ again represent probability distributions.

The following is the algorithm to calculate the weights for context-independent linear interpolation λ₁, λ₂, λ₃ when
the $n$-gram frequencies are known. N is the size of the corpus. If the denominator in one of the expressions
is 0, we define the result of that expression to be 0. _**[Page 3 of ref [3](https://aclanthology.org/A00-1031.pdf)]**_

```
Set λ₁ = λ₂ = λ₃ = 0

For each trigram t₁, t₂, t₃ with f(t₁, t₂, t₃) > 0:
    Depending on the maximum of the following three values:
    
    Case f(t₁, t₂, t₃) - 1 / f(t₁, t₂) - 1:
        Increment λ₃ by f(t₁, t₂, t₃)
    
    Case f(t₂, t₃) - 1 / f(t₂) - 1:
        Increment λ₂ by f(t₁, t₂, t₃)
    
    Case f(t₃) - 1 / N - 1:
        Increment λ₁ by f(t₁, t₂, t₃)
        
Normalize λ₁, λ₂, λ₃
```

This idea can be easily extrapolated to any $n$-gram, by simply considering the $n$-gram probability distribution as
being
dependent on all the previous $i$-gram's Maximum Likelihood Estimates (MLEs) (where $1 \leq i \leq n$).

---

# Generation

## Instructions to run

```python3 generator.py <N> <lm_type> <corpus_path> <k> <gen_type>```  
Here, `<lm_type>` must be one of `l` for Laplace Smoothing, `g` for Good-Turing Smoothing, `i` for Linear
Interpolation, 'n' for No Smoothing.  
`<N>` is the $N$-gram size, `<corpus_path>` is the path to the corpus, and `<gen_type>` must be one of `w` for Next Word
generation (in which case, `<k>` is number of candidates for next word to be printed) or 's' for Sentence generation (in
which case, `<k>` is number of next words to be generated in the sentence.)

For sentence generation, the most probable word is chosen at each stage to form the sentence, until either `k` words
are generated or the end of sentence token `<\s>` is generated.

---

# Analysis

Please refer to the [report](Report.md) for an analysis of these models.

---

