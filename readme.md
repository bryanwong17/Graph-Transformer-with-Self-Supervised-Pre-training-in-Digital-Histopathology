# Graph Transformer with Self-Supervised Pre-training in Digital Histopathology Documentation

1. Patch Tiling

**How to run:**
- python patch_tiler.py

**Note:**
- check arg parser for the configurations


2. Clustering (optional)

**How to run:**
- python preprocessing/clustering.py

**Note:**
- number of clusters is equal to batch size (32)


3. Training Patch Feature Extractor

**How to run:**

- python feature_extractor/feature_extract.py

**Notes:**

- check config.yaml for configuration
- change either want to use SimCLR, SimSiam or MoCoV3 in feature_extract.py (simply comment/uncomment it)
- if want to use Pytorch Lightning, run feature_extract_lightning.py instead (only MoCoV3 is ready)
- model path: ../../graph_transformer/runs/(simclr or mocov3)/runs/(name of SSL model)


4. Build Graph

**How to run:**

- python feature_extractor/build_graphs.py

**Notes:**

- check arg parser for configuration
- choose between loading SimCLR, SimSiam, and MoCoV3 model in build_graphs.py (simply comment/uncomment it)
- graph path: ../../graph_transformer/build_graphs/(simclr, mocov3, or simsiam)/(name of graph)


5. Training Graph Transformer

**How to run:**

- cd ..
- python main.py (check arg parser at option.py)

**Notes:**
- graph path: ../graph_transformer/build_graphs/(simclr, mocov3, or simsiam)/(name of graph)/(simclr_files or simsiam_files)
- when testing: train=False, test=True, val_set=(test data), resume=(graph VIT model path)
- result path: ../graph_transformer/results_with_graph/(simclr, mocov3, or simsiam)/(name of result folder)


6. Training without graph (VIT)

**How to run:**

- change load model in main.py to "from models.OnlyVisionTransformer import Classifier"
- python main.py (check arg parser at option.py)

**Note:**
- result path: ../graph_transformer/results_without_graph/(simclr, mocov3, or simsiam)/(name of result folder)