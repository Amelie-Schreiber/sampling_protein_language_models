# Variations of Gibbs Sampling of Protein Language Models for Directed Evolution in Silico

This repo uses variations of Gibbs sampling of masked (protein) language models, in particular, ESM-2, in order to sample protein sequence space 
to create novel proteins. It can be used in conjunction with various scoring methods such as the masked marginal scoring function to form a directed 
evolution graph with nodes representing proteins, and directed edges indicating fitness of the mutation as scored by the protein language model. 
This can be used to get a sense of the fitness landscape of mutated proteins. 

For example, one can generate multiple mutated proteins from the Gibbs sampling functions, then, using the masked marginal scoring one can determine 
if the mutations improve or degrade the fitness of the protein to obtain directed edges. One can then sample using the generated proteins to expand 
the directed graph further. 

---

Let's break down the function `gibbs_sampling_with_random_masking` and provide a rigorous mathematical explanation of its steps.

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
\mathbf{tokenizer} = \mathbf{AutoTokenizer.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

$$
\mathbf{model} = \mathbf{EsmForMaskedLM.from_pretrained("facebook/esm2_t6_8M_UR50D")}
$$

These are responsible for converting our sequence into a format the model can understand and for making predictions, respectively.

#### 3. Gibbs Sampling Iteration:

For each iteration, ranging from 1 to a predefined number $T$, we perform the following steps:

#### 3.1 Tokenization:

The protein sequence $S$ is tokenized into a sequence of token IDs $T(S)$ of length $L$ using:

$$
T(S) = \mathbf{tokenizer}(S, \mathbf{return_tensors}="pt")
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
\mathbf{logits} = \mathbf{model}(\mathbf{input_ids} = T'(S)).\mathbf{logits}
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
S = \mathbf{tokenizer.decode}(T(S)[0], \mathbf{skip\_ special\_ tokens=True})
$$

This updated sequence $S$ is then used in the next iteration.

#### 4. Conclusion:

After $T$ iterations, the function returns the final modified sequence $S$.

### Final Remarks:

The process described uses a combination of random masking and a masked language model to iteratively modify and potentially refine a protein sequence. This stochastic process allows for the exploration of sequence space, enabling the discovery of new and plausible sequences that are consistent with known protein patterns.
