# Text-Enhanced Representation Learning for Knowledge Graph
## Key Idea  
By regarding each relation as one translation from head entity to tail entity, existed translation-based methods including TransE, TransH and TransR are simple, effective and achieving the state-of-the-art performance. However, they still suffer the following issues:  
* low performance when modeling 1-to-N, N-to-1 and Nto-N relations.  
* limited performance due to the structure sparseness of the knowledge graph.  
This paper takes a text corpus as input and attempt to incorporating deep contextual information to the KG. This paper also enable each relation to own different representations for different head and tail entities, which is proved to be helpful to handle the low performance on 1-to-N, N-to-1 and N-to-N relations.  

## How?
### Incorporating deep contextual information to the KG
* STEP1: Generating text corpus from the English Wikipedia. For FreeBase, this paper focuses on the Wikipedia inner links and automatically annotate the links as the Freebase entities if the linked Wikipedia entities have the same titles as the Freebase entities, otherwise as the lexical words. For WordNet, this paper ignores the Wikipedia links and annotate the words as the WordNet entities if the words belong to the WordNet synsets. And this paper trains the skip-gram word2vec model on the entity-annotated texts.  
* STEP2: Constructing a co-occurrence network $G = (X , Y)$ based on the entity-annotated text corpus.  $x_i\in X$ denotes the node of the network and corresponds to a word or an entity. $y_{ij}\in Y$ represents the co-occurrence frequency between $x_i$ and $x_j$.  
* STEP3: something are defined as follow:  
***pointwise textual context***: $n(x_i) = \{ x_j | y_{ij} > \theta \} $, where $\theta$ is the threshold and the neighboring nodes whose co-occurrence frequencies are lower than  $\theta$ are filtered.  
***pairwise textual context***: $n(x_i, x_j) = \{ x_k|x_k \in n(x_i) \bigcap n(x_j) \} $  
***pointwise textual context embedding***: $$**n**(x_i) $$
