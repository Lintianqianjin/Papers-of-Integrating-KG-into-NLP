# Locally Adaptive Translation for Knowledge Graph Embedding  

## Key Idea  
Existing embedding methods over one knowledge graph determine the optimal form of loss function during experiments. A critical problem is that the determination is made over a limited number of candidates. It is unclear that why the loss function is examined by ***only testing a closed set of values of its parameters***(optimal margin in TransE and other similiar methods).  
  Existing embedding methods over two different knowledge graphs find their individual optimal loss functions over the same set of candidates. Since different knowledge graphs contain different entities and relations, this compatible setting ***ignores the individual locality of knowledge graphs*** and seems not convincing in theory.  
  TransA which thie paper proposed ***adaptively finds the optimal loss function according to the structure of knowledge graphs***, and no closed set of candidates is needed in advance.  
 
### How to adaptively choose the optimal margin  
Classical knowledge graph is fully made up of two disjoint sets, i.e., the entity set and relation set, it makes sense that the optimal margin, denoted by $M_{opt}$, is composed of two parts, namely, ***entity-specific margin*** $M_{ent}$, and ***relation-specific*** margin $M_{rel}$. Furthermore, it is natural to linearly combine the two specific margins via a parameter $\mu $ which controls the trade-off between them. Therefore, the optimal margin of embedding satisfies $$M_{opt} = \mu M_{ent} + (1 − \mu )M_{rel},$$ where $0 \leq \mu \leq 1$.To demonstrate that the margin $M_{opt}$ is optimal, it is sufficient to find the optimal entity-specific margin and the optimal relation-specific margin.  

#### How to adapt $M_{ent}$ and $M_{rel}$
##### $M_{ent}$
The positive entities have the same relation with $h$ (or $t$), and the negative entities have different relations with $h$ (or $t$). In this sense, the optimal margin $M_{ent}$ is actually equal to ***the distance between two concentric spheres in the vector space***, illustrated as figure below. The positive entities (illustrated as “⚪”) are constrained within the internal sphere, while the negative entities (illustrated as “□”) lie outside the external sphere.  
![Ment.png](https://github.com/Lintianqianjin/Papers-of-Integrating-KG-into-NLP/blob/master/AAAI/2016/Locally%20Adaptive%20Translation%20for%20KG%20Embedding/Ment.png)

##### $M_{rel}$
Given a specific entity $h$ and one of its related relation $r$, other relations with $h$ as one end have different degrees of similarity with the relation $r$. To measure this similarity, the length of relation-specific embedding vectors can be considered. Relations except $r$ can be classified into two parts according to whether its length is larger than $\| \| r\| \| $. For relations $r_i$, $r_j$, we assume that $r_i$ is more similar with $r$ than $r_j$ if $\| \| r_i\| \| − \| \| r\| \| \leq \| \| r_j\| \| − \| \| r\| \|$. Then similar to the analysis of entity-specific margin, the optimal relation-specific margin is equal to ***the distance between two concentric spheres in the vector space***. The internal sphere constraints relation $r$ and those with length smaller than $\| \| r\| \| $, while the relations with length greater than $\| \| r\| \| $ lie outside the external sphere.   
![Mrel.png](https://github.com/Lintianqianjin/Papers-of-Integrating-KG-into-NLP/blob/master/AAAI/2016/Locally%20Adaptive%20Translation%20for%20KG%20Embedding/Mrel.png)

## Question
Definition of entity-specific margin seems to be wrong.  
The amount of concentric spheres in the vector space when it comes to relation-specific margin seems to be there in most situations.   
