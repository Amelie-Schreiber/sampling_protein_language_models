# sampling_protein_language_models
Using variations on Gibbs sampling for directed evolution

## How the Functions Work

Let's break down the function `gibbs_sampling_likeliest_with_random_masking` and provide a rigorous mathematical explanation of its steps.

### Mathematical Explanation of the Gibbs Sampling with Random Masking Function

#### 1. Problem Definition:

Given a protein sequence $S$ of length $L$ defined as:

$$
S = \{s_1, s_2, ..., s_L\}
$$

where each $s_i$ represents a token (an amino acid or special token), the goal is to modify the sequence through a stochastic process to explore the potential sequence space.

#### 2. Initialization:

Firstly, we initialize the `tokenizer` and the `model`:

$$
\text{tokenizer} = \text{AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

$$
\text{model} = \text{EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

These are responsible for converting our sequence into a format the model can understand and for making predictions, respectively.

#### 3. Gibbs Sampling Iteration:

For each iteration, ranging from 1 to a predefined number $T$, we perform the following steps:

#### 3.1 Tokenization:

The protein sequence $S$ is tokenized into a sequence of token IDs $T(S)$ of length $L$ using:

$$
T(S) = \text{tokenizer}(S, \text{return_tensors}="pt")
$$

#### 3.2 Masking:

For each iteration, a percentage $p$ of the total tokens $L$ are randomly selected for masking. The number of tokens to mask is given by:

$$
m = \lceil p \times L \rceil
$$

Let $M$ be the set of positions selected for masking. Then, for each position $i$ in $M$, we replace the token at that position with the special "mask" token, resulting in a new token sequence $T'(S)$.

#### 3.3 Model Prediction:

With the masked sequence $T'(S)$, we feed it into our pre-trained model to get a prediction for the masked tokens:

$$
\text{logits} = \text{model}(\text{input_ids} = T'(S)).\text{logits}
$$

The `logits` are raw prediction scores for each token in the vocabulary at the masked positions.

#### 3.4 Token Replacement:

For each masked position $i$ in $M$, we identify the token ID with the highest logit score:

$$
\text{predicted\_token\_id} = \arg\max_{j} \text{logits}[0, i, j]
$$

This ID represents the most likely token (according to the model) to replace the masked token. We then update the original token sequence \( T(S) \) at position \( i \) with the `predicted_token_id`.

#### 3.5 Sequence Update:

After all masked positions in $M$ have been replaced by their respective predicted tokens, we decode the updated token sequence back into a protein sequence format:

$$
S = \text{tokenizer.decode}(T(S)[0], \text{skip\_special\_tokens=True})
$$

This updated sequence $S$ is then used in the next iteration.

#### 4. Conclusion:

After $T$ iterations, the function returns the final modified sequence $S$.

### Final Remarks:

---

The process described uses a combination of random masking and a masked language model to iteratively modify and potentially refine a protein sequence. This stochastic process allows for the exploration of sequence space, enabling the discovery of new and plausible sequences that are consistent with known protein patterns.

Let's break down the modified function `gibbs_sampling_from_distribution_with_random_masking` and provide a rigorous mathematical explanation of its steps.

### Mathematical Explanation of the Gibbs Sampling with Random Masking Function (Probabilistic Replacement)

#### 1. Problem Definition:

Given a protein sequence $S$ of length $L$ defined as:

$$
S = \{s_1, s_2, ..., s_L\}
$$

where each $s_i$ represents a token (an amino acid or special token), the goal is to modify the sequence through a stochastic process to explore potential sequence space.

#### 2. Initialization:

Firstly, we initialize the `tokenizer` and the `model`:

$$
\text{tokenizer} = \text{AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

$$
\text{model} = \text{EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

These are responsible for converting our sequence into a format the model can understand and for making predictions, respectively.

#### 3. Gibbs Sampling Iteration:

For each iteration, ranging from 1 to a predefined number $T$, we perform the following steps:

#### 3.1 Tokenization:

The protein sequence $S$ is tokenized into a sequence of token IDs $T(S)$ of length $L$ using:

$$
T(S) = \text{tokenizer}(S, \text{return_tensors}="pt")
$$

#### 3.2 Masking:

For each iteration, a percentage $p$ of the total tokens $L$ are randomly selected for masking. The number of tokens to mask is given by:

$$
m = \lceil p \times L \rceil
$$

Let $M$ be the set of positions selected for masking. Then, for each position $i$ in $M$, we replace the token at that position with the special "mask" token, resulting in a new token sequence $T'(S)$.

#### 3.3 Model Prediction:

With the masked sequence $T'(S)$, we feed it into our pre-trained model to get a prediction for the masked tokens:

$$
\text{logits} = \text{model}(\text{input_ids} = T'(S)).\text{logits}
$$

The `logits` are raw prediction scores for each token in the vocabulary at the masked positions.

#### 3.4 Convert Logits to Probabilities:

To draw samples based on the predicted token probabilities, we first convert the logits for each masked position \( i \) to a probability distribution using the softmax function:

$$
P(s_i) = \text{softmax}(\text{logits}[0, i])
$$

Where $P(s_i)$ is a probability distribution over all possible tokens for position $i$.

#### 3.5 Probabilistic Token Replacement:

For each masked position $i$ in $M$, we draw a token based on the softmax probability distribution:

$$
\text{predicted\_token\_id} = \text{sample}(P(s_i))
$$

Here, the `sample` function represents drawing a single token ID based on the probabilities $P(s_i)$. This ensures that the replacement is probabilistic and not just the token with the highest likelihood.

We then update the original token sequence $T(S)$ at position $i$ with the `predicted_token_id`.

#### 3.6 Sequence Update:

After all masked positions in $M$ have been replaced by their respective predicted tokens, we decode the updated token sequence back into a protein sequence format:

$$
S = \text{tokenizer.decode}(T(S)[0], \text{skip\_special\_tokens=True}).\text{replace}(" ", "")
$$

This updated sequence $S$ is then used in the next iteration.

#### 4. Conclusion:

After $T$ iterations, the function returns the final modified sequence $S$.

### Final Remarks:

The modified version of the Gibbs sampling function introduces a significant change in the replacement strategy. Instead of deterministically replacing a masked position with its most likely token, the function uses a probabilistic approach, sampling from the predicted distribution of tokens. This ensures a richer exploration of the sequence space, allowing for diverse sequences that are consistent with known protein patterns.
