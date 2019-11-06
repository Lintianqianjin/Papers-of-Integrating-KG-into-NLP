# Incorporating Relational Knowledge into Word Representations using Subspace Regularization  
## Key Idea  
In modeling relational knowledge, existed methods make a rather restrictive assumption requiring all triplets $\(w_i,r,w_j\)$ pertaining to a relation type $r$ to satisfy $w_i + r \approx w_j$. This restriction can be severe when learing from a large text corpus since vector representation of a word also needs to respect a huge set of co-occurence instances with other words. This restriction is also not suitable for modeling **symmetric relations** and **transitive relations**.  
This paper proposed a novel formulation for modeling the relational knowledge which addresses these issues by relaxing the $w_i + r \approx w_j$ and modeling each relation by a low-rank subspace, i.e., **all the word pairs pertaining to a relation are assumed to lie in a low-rank subspace**.  

## Subspace-regularized word embedding  
$R_k = \{\(w_i, r_k, w_j\)\} \forall1 1 \leq k \leq m$, where words $w_i$ and $w_j$ are connected by relation $r_k$ and $R_k$ is the set of all triplets corresponding to relation rk with |Rk| = nk.  
Let $d_{ij} = \(w_j − w_i\)$ denote the **difference vector** for the triplet $(w_i, r_k, w_j)$ which points from the vector of word $w_i$ to that of word $w_j$. And matrix $D_k$ is stacking the difference vectors corresponding to all the triplets in relation $r_k$, i.e.,  
$$D_k = \[\cdot\cdot\cdot d_ij\cdot\cdot\cdot\] \forall \{(i,j): \(w_i,r_k,w_j\)\in R_k\}$$.  
To incorporate this relational knowledge into word embeddings, this paper enforces an approximate low-rank constraint on $D_k$ assuming $D_k\approx U_k{A_k}^T$.