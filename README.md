HGMG: Enhancing heterogeneous graph embedding by meta-graph learning
This repository contains the implementation of HGMG (Heterogeneous Graph Meta-Graph Model), 
a deep learning framework for heterogeneous graph representation learning and node classification. 
The model integrates multiple meta-graphs, instance-level embeddings, 
and similarity-based graph neural networks to capture complex structural information in heterogeneous graphs.

Overview
Meta-graph filtering based on meta-graph structure：Metagraph_Select.py
Node embedding based on meta-graph context:Node_emb_model.py
Node embedding enhancement based on meta-graph instance:Instance_emb_model.py
Meta-Graph Fusion:Metagraph_fusion
Node embedding based on node similarity：SimNode_emb_model.py

If you want to get the meta graph structure, please first download the original GraMi code from 
[https://github.com/ehab-abdelhamid/GraMi](https://smufang.github.io/code/GraMi%20with%20automorphism.zip).
