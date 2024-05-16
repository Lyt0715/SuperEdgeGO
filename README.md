# Learning Better Representation for Protein Function Prediction: Encoding Residue Contacts by Supervised Graph Attention
## The overall architecture of SuperEdgeGO
![The overall architecture of SuperEdgeGO](https://github.com/Lyt0715/SuperEdgeGO/blob/main/Fig_1.png)

## Adjacency matrix and feature matrix
The input protein sequence is first sent to the protein language model ESM-2 to generate the feature matrix, and to the protein structure model AlphaFold2 to predict structures, which is eventually processed as the adjacency matrix.
