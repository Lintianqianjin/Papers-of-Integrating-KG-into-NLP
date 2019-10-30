# TransG : A Generative Model for Knowledge Graph Embedding
## Key Idea
### multiple realtion semantics
TransE adopts the principle of $t − h \approx r$, there is supposed to be only one cluster whose centre is the relation vector r. However, results show that there exist multiple clusters.  
This paper leverages a **Bayesian non-parametric infinite mixture model** to handle multiple relation semantics by generating multiple translation components for a relation.  
