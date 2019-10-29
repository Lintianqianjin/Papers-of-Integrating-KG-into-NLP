# Jointly Embedding Knowledge Graphs and Logical Rules  

---

## Key Idea  
Most existing methods perform the embedding
task based solely on fact triples. Logical rules, although containing rich background information, have not been well studied in this
task. This paper proposes a novel method of
jointly embedding knowledge graphs and logical rules.  

## Defined Symbol  
$e_i$: vector of entity $i$  
$r_k$: vector of relation $k$  
$(e_i,r_k,e_j)$: when this triple holds, $e_i+r_k \approx e_j$, e.g. $paris+capitalOf = France$  
$I(e_i,r_k,e_j) = 1- {1} \over {3\sqrt d} ||e_i+r_k-e_j||_1$: Correct probability of this triple.