# Variations of Gibbs Sampling of Protein Language Models for Directed Evolution in Silico

This repo uses variations of Gibbs sampling of masked (protein) language models, in particular, ESM-2, in order to sample protein sequence space 
to create novel proteins. It can be used in conjunction with various scoring methods such as the masked marginal scoring function to form a directed 
evolution graph with nodes representing proteins, and directed edges indicating fitness of the mutation as scored by the protein language model. 
This can be used to get a sense of the fitness landscape of mutated proteins. 

For example, one can generate multiple mutated proteins from the Gibbs sampling functions, then, using the masked marginal scoring one can determine 
if the mutations improve or degrade the fitness of the protein to obtain directed edges. One can then sample using the generated proteins to expand 
the directed graph further. 
