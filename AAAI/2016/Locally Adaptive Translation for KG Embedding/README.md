# Locally Adaptive Translation for Knowledge Graph Embedding  

## Key Idea  
Existing embedding methods over one knowledge graph determine the optimal form of loss function during experiments. A critical problem is that the determination is made over a limited number of candidates. It is unclear that why the loss function is examined by ***only testing a closed set of values of its parameters***(optimal margin in TransE and other similiar methods).  
  Existing embedding methods over two different knowledge graphs find their individual optimal loss functions over the same set of candidates. Since different knowledge graphs contain different entities and relations, this compatible setting ***ignores the individual locality of knowledge graphs*** and seems not convincing in theory.  
  TransA which thie paper proposed ***adaptively finds the optimal loss function according to the structure of knowledge graphs***, and no closed set of candidates is needed in advance.  
 
### How to adaptively choose the optimal margin  
Classical knowledge graph is fully made up of two disjoint sets, i.e., the entity set and relation set, it makes sense that the optimal margin, denoted by $M_{opt}$, is composed of two parts, namely, entity-specific margin $M_{ent}$, and relation-specific margin $M_{rel}$. Furthermore, it is natural to linearly combine the two specific margins via a parameter $\mu $ which controls the trade-off between them. Therefore, the optimal margin of embedding satisfies $$M_{opt} = \mu M_{ent} + (1 − \mu )M_{rel},$$ where $0 \leq \mu \leq 1$.To demonstrate that the margin $M_{opt}$ is optimal, it is sufficient to find the optimal entity-specific margin and the optimal relation-specific margin.  

