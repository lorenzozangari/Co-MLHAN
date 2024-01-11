# Co-MLHAN
Implementation of Co-MLHAN: contrastive learning for multilayer heterogeneous attributed networks, as presented in our paper:
```
Martirano, L., Zangari, L. & Tagarelli, A.
Co-MLHAN: contrastive learning for multilayer heterogeneous attributed networks. 
Appl Netw Sci 7, 65 (2022). https://doi.org/10.1007/s41109-022-00504-9
```

>Our proposed Co-MLHAN is a self-supervised  graph representation 
learning approach conceived for  multilayer 
heterogeneous  attributed networks. A key novelty of Co-MLHAN is its higher expressiveness w.r.t. existing methods, since  heterogeneity is assumed to hold at both  node and edge levels,  possibly for  each layer of the network. 

## Requirements
The dependendencies are as follows:

- python >= 3.7
- torch >= 1.13.1
- numpy
- scikit-learn
- pandas
- dgl == 1.1.2


## Data loading
Co-MLHAN and Co-MLHAN-SA require the following text files:
- nodes.txt: each line represents a node, and contains three tokens <layer\> <node_type\> <node\>, where node_type is a string (a single character uniquely identifying a node type), and all the other tokens are in a numeric format. The progressive node IDs start from 0 for each type. Make sure to provide instances of the same entity in different layers with the same ID.
- edges.txt: each line represents an intra-layer edge, and contains four tokens <layer\> <edge_type\> <node1\> <node2\>, where edge_type is a string (concatenation of node types comprising the edge type (base_type), plus a possible suffix if different types of edges exist and must be distinguished between the same types of nodes), and all other tokens are in a numeric format. If the suffix is required, it is a single character, unique for all edge types (not just those between nodes of the same type), following the underscore "_" after the base_type. The ordering <node1\> <node2\> matches the base_type. We recall that the network schema discards all edges not containing the target type in the base_type (i.e., all lines where neither <node1\> nor <node2\> are of target type).
-edges_across.txt (optional): each line represents an inter-layer edge, and contains five tokens <layer1\> <layer2\> <edge_type\> <node1\> <node2\>, where edge_type is a string (concatenation of node types comprising the edge type (base_type), plus a possible suffix if different types of edges exist and must be distinguished between the same types of nodes), and all other tokens are in a numeric format. The ordering <node1\> <node2\> matches the <edge_type\>. We specify that <node1\> belongs to <layer1\> and <node2\> belongs to <layer2\>. We recall that the network schema discards all edges not containing the target type in the base_type (i.e., all lines where neither <node1\> nor <node2\> are of target type).
- metapath_type.txt (one for each meta-path type, named as the meta-path type, i.e., the concatenation of node types comprising the meta-path type, plus possible suffixes if the corresponding edges are crossed, in the same order as they are crossed): each line represents a pair of target nodes connected by at least one meta-path instance of such type, and contains four tokens: <layer\> <node_s\> <node_t\> <count\>, where each token is in a numeric format. <node_s\> is the starting node (of target type), <node_t\> is the terminal node (of target type), and <count\> is the number of meta-path instances of such type connecting <node_s\> and <node_t\> in layer <layer\>. Since the resulting meta-path based graph is oriented, make sure to provide both directions of meta-path instances.
- metapath_type_across.txt (one for each meta-path type, for Co-MLHAN only, named as the meta-path type (i.e., the concatenation of node types comprising the meta-path type), plus possible suffixes if the corresponding edges are crossed, in the same order as they are crossed, plus the '_across' suffix): each line represents a pair of target nodes connected by at least one meta-path instance of such type, and contains five tokens <layer_s\> <layer_t\> <node_s\> <node_t\> <count\>, where each token is in a numeric format. <layer_s\> is the layer containing <node_s\>, <layer_t\> is the layer containing <node_t\>, <node_s\> is the starting node (of target type), <node_t\> is the terminal node (of target type), aand <count\> is the number of meta-path instances of such type connecting <node_s\> in layer <layer_s\> and <node_t\> in layer <layer_t\>. We recall that the intermediate node matches a pillar-edges, i.e., exists in both <layer_s\> and <layer_t\>.
- positives.txt: first line specifies X meta-path types, where X is the number of (within layer) meta-path types used for the count of positives. From the second line, each line represents the meta-paths count given a pair of nodes, and contains two + X tokens <node1\> <node2\> <c1\> ... <cX\>, where each token is in a numeric format and each c is the count of the meta-path instances of that type connecting <node1\> and <node2\> in any layer.
- features_EL.txt (optional): each line represents an entity, and contains two + dim tokens <node_type\> <node\> <f1\> <f2\> ... <fdim\>, where node_type is a string (a single character uniquely identifying the type), all the other tokens are in a numeric format, and dim is the embedding dimension (default 64). Note that features_EL.txt is an alternative to features_NL.txt.
- features_NL.txt (optional): each line represents a node, and contains three + dim tokens <layer\> <node_type\> <node\> <f1\> <f2\> ... <fdim\>, where node_type is a string (a single character uniquely identifying the type), all the other tokens are in a numeric format, and dim is the embeddings dimension (default 64). Note that features_NL.txt is an alternative to features_EL.txt.
  
For the final downstream task, you should also provide additional files. See the evaluation section for further detail.

All these files must be placed in "data/<dataset_name>/", where <dataset_name\> is the name of the dataset.

### Pre-processing

Before training the model a pre-processing step is required. You should run the script *input_data\data_preprocessing.py*, specifying the following parameters:

- *dataset*, string indicating the name of the dataset (e.g., --dataset *imdb_mlh*)
- *metapath*, string indicating each meta-path type separated by ; (e.g., --metapath *MAM;MDM*)
- *target*, string indicating the target type (e.g., --target *M*)
- *layers*, integer value indicating the number of layers
- *pos_th*, integer value indicating the threshold for positives (T_pos).
-*pos_cond*, string indicating the condition for the positives construction. That is, an integer value for each meta-path must be given, separated by comma.
The final condition  is placed in OR between the meta-path provided into the file positives.txt. For example, --pos_cond *3,1*, indicates 3 MAM OR 1 MDM (assuming that
  in the file positives.txt, the meta-path order is MAM and MDM). Indeed, the order of the conditions must follow the order of the meta-paths specified in the first line of the positives.txt file
- *features*, string indicating the name of features file (e.g., --features *features_EL*)
- *node_lf*, information needed when features are at node-level. Do not specify it, if features are at entity-level (see the paper for further detail).

This step will create a new folder (*"data/<dataset_name\>/prep_data"*) where the artifacts for running CO-MLHAN are saved.

## Train the model

For training CO-MLHAN, run the script train.py specifying the following parameters:

- *sa*, specify this parameter if you want to use 'Co-MLHAN-SA'.
- *target*, uppercase character indicating target type (e.g., 'M').
- *layers*, integer indicating the number of layers of the multilayer graph (default is 2).
- *metapath*, string indicating all the metapath types, separated by comma (e.g., 'MAM,MDM').
- *node_lf*, specify this parameter if you are employing node level features. Note that, this specification
  must be consistent with "features" parameter (see below).
- *features*, string indicating the features for each entity, e.g., "M,featuresNL;A;featuresA;D,featuresD". If features
  are missing you can also specify to use identity matrix or a probability distribution (e.g., "M;identity;A,normal;D,uniform").
- *hidden_dim*, integer indicating the dimension of hidden layers (default is 64).
- *num_hidden*, number of hidden layers (default is 1).

- *epochs*, number of epochs (default is 10000).
- *patience*, patience value for early stopping (default is 30).
- *lr*, learning rate (default is 0.0001).
- *l2_coef*, coefficient for L2 regularization (default = .0).
- *tau*, temperature value (default = 0.5).
- *feat_drop*, dropout value for hidden states (default = 0.3).
- *attn_drop*, dropout attention (default is 0.3).
- *lam*, balancing coefficient between network schema and meta-path view (default is 0.5).
- *n_heads*, number of attention heads (default is 1).
- *rates*, string indicating the neighbor sampling rate for each edge type. For example,
  "MA,7;MD,2" indicates that during the training, 7 MA and 2 MD edges are sampled for each target type. The sampling
  can be with or without replacement (see no_replace below).
- *no_replace*, whether to sample neighbors without replacement.
- *ncl*, whether to avoid across-layer meta-path

- *dataset*, dataset name.
- *gpu*, which gpu to use.
- *seed*, integer specifying the seed value.
- *save_emb*, whether to save the final embedding.
- *checkpoint*, directory for saving checkpoint models (for early stopping).

### Example: training on imdb_mlh dataset
```
python train.py --dataset imdb_mlh --target M --metapath MAM,MDM --layers 2 --features 'M,features1000;A,identity;D,identity'
--rates MA,7;MD,2 --epochs 10000 --attn_drop 0.3 --save_emb
```

We provided two subset from IMDB, named **imdb_mlh**, and **imdb_mlh_mb**. See our paper
for further detail about these datasets.

## Evaluation

You can train the final embedding for solving several tasks, such as entity classification, link
prediction, clustering, etc, by providing your own predictor. We employed a multilayer
perceptron for evaluating the embedding on the entity classification task. Further details
are provided in our paper.

## Reference
If you use this code, or the data we provided, please cite our paper:

```
Martirano, L., Zangari, L. & Tagarelli, A.
Co-MLHAN: contrastive learning for multilayer heterogeneous attributed networks. 
Appl Netw Sci 7, 65 (2022). https://doi.org/10.1007/s41109-022-00504-9
```

