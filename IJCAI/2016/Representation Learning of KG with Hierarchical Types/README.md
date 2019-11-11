# Representation Learning of Knowledge Graphs with Hierarchical Types  

## Key Idea  
Conventional methods merely focus on the structured information in triples, paying less attention to the rich information located in hierarchical types of entities.  For instance, *William Shakespeare* has a variety of types (e.g. *book/author*, *award/award nominee* and *music/artist*) and shows different attributes under different types. It's strongly believed that every entity in different scenarios, as the reflections of itself from various angles, should have different representations.  

## How?  
### Hierarchical type structure defined  
Taking a hierarchical type $c$ with $k$ layers for instance, $c^{(i)}$ is the *i*-th sub-type of $c$. The most precise subtype is the first layer and the most general sub-type is the last layer, while each sub-type $c^{(i)}$ has only one parent sub-type $c_{(i+1)}$. Walking through the bottom-up path in hierarchical structure, we can get the representation of hierarchical type as $c = \{ c^{(1)}, c^{(2)}, ..., c^{(k)}\}$.  
### Projection matrix for type $c$ construction  
Projection matrix for type $c$, denoted by $M_c$, can be constructed in two ways, ***Recursive Hierarchy Encoder*** and ***Weighted Hierarchy Encoder***.  
#### Recursive Hierarchy Encoder
$$M_c = \prod_{i=1}^m M_{c^{(i)}} = M_{c^{(1)}}M_{c^{(2)}}...M_{c^{(m)}}$$
where $m$ is the number of layers for type $c$ in the hierarchical structure, while $M_{c^{(i)}}$ represents the projection matrix of the *i*-th sub-type $c^(i)$.  
#### Weighted Hierarchy Encoder
$$M_c = \sum_{i=1}^m \beta_i M_{c^{(i)}} = \beta_1 M_{c^{(1)}} + \beta_2 M_{c^{(2)}} +...+ \beta_m M_{c^{(m)}}$$
where $\beta_i$ is the corresponding weight of $c^{(i)}$, and this paper designs a proportional-declined weighting strategy between $c^{(i)}$ and $c^{(i+1)}$ as follows:$$\beta_i : \beta_{i+1} = 1-\eta : \eta , \sum_{i=1}^m \beta_i = 1, \eta \in (0,0.5).$$ The strategy indicates that the more precise sub-type $c^{(i)}$ is, the higher weight $\beta_i$ will be, thus the greater influence $c^{(i)}$ will have on $M_c$.  

Then, $M_{rh}$ and $M_{rt}$ are defined as follow:
$$M_{rh} = {\sum_{i=1}^n \alpha_i M_{c_i}}\over {\sum_{i=1}^n \alpha_i}, \alpha_i = \begin{cases} 1& \text{c_i \in C_{rh}} \\ 0& \text{c_i \notin C_{rh}}\end{cases}$$
