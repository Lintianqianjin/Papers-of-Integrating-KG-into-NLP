# A Joint Model for Question Answering over Multiple Knowledge Bases
## Key Idea  
The most significant distinction between multiple KB-QA and single KB-QA is that the former must consider the alignments between KBs. It is natural to adopt a pipeline strategy including two steps: alignment construction and query construction. First, the alignments between heterogeneous KBs are identified. Then, multiple KBs could be linked together and be regarded as a large single KB. Thus, existing single KB-QA methods could be applied. However, such pipeline strategy has two limitations.  
* The alignments obtained by automatic methods are inevitably noisy and such noises would be passed on to the subsequent step and have negative effects on the final answer generation.  
* Existing KBs usually grow fast and update frequently, the constructed alignments are prone to be out of date and new alignments need to be added.  

Therefore, this paper ***perform alignment construction and query construction jointly*** to alleviate these problems. This paper presents a novel joint model based on integer linear programming (ILP), uniting these two procedures into a uniform framework.  

## How?  
The method has five steps as follows.   
* The first step: phrase detection & resource mapping(computing similarity based on Levenshtein distance between label of a resource and the word sequence).  
* The second step: candidate triple pattern generation.  
* The third step: potential alignment generation(if two entities from different KBs are similar enough, they are identified as an alignment.).  
* The fourth step: global joint inference(more important triple patterns and more reliable alignments should be selected. this paper also considers that the selected triple patterns should cover as many as words in the input question.).  
* The final step: formal query generation.  

## Question
This paper said the method only cost a few seconds to solve those problems, however, time complexity of this method is at least $O(nlog_2n)$.   
