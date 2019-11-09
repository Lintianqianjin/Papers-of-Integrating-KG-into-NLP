# Knowledge Representation Learning with Entities, Attributes and Relations  
## Key Idea
Some relations indicate attributes of entities, and others indicate relations between entities. Former is the primary source of one-to-many and many-to-one relations. Hence, existing KG-relations can be divided into ***attributes*** and ***relations***.

## How? 
Firstly, this paper manually divides the original Freebase relations into two types: attributes and relations.  
* For attributes, a score function $h()$ is defined as follows. Entity embeddings are transformed into the attribute space via a single-layer neural network, and then calculate the semantic similarity between the transformed embedding and the embedding of the corresponding attribute value:$$h(e,a,v) = -\| \| f(eW_a + b_a) - V_{av} \| \| + b_2$$ where $e$, $a$, $v$ are entity, attribute relation, attribute property, respectively. $f()$ is a nonlinear function such as $tanh$, $V_{av}$ is the embedding of attribute value $v$ and $b_2$ is a bias constant.  
* For relations, the score function can be one of those score functions in previous paper, e.g., TranE and TransR.  
