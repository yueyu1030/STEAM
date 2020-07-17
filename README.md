# STEAM
This is the code repository for our KDD'20 paper STEAM: Self-Supervised Taxonomy Expansion with Mini-Paths.

## Requirements
* Python >= 3.6 
* PyTorch >= 1.2
* tqdm 
* Scipy
* Numpy
* transformers
<!-- ## Features
* Clear folder structure which is suitable for many deep learning projects.
* `.json` config file support for convenient parameter tuning.
* Customizable command line options for more convenient parameter tuning.
* Checkpoint saving and resuming.
* Abstract base classes for faster development:
  * `BaseTrainer` handles checkpoint saving/resuming, training process logging, and more.
  * `BaseDataLoader` handles batch generation, data shuffling, and validation data splitting.
  * `BaseModel` provides basic model summary. -->
## Folder Structure
  ```
  ├── model/ - models, losses, and metrics
  │   ├── model_fuse.py // main modules of STEAM
  │   ├── layers_path.py // neural layers of STEAM
  │   ├── run_fuse.sh // script to run the code
  │   ├── utils_path.py // utility functions: loading train data, test data and sample mini-paths
  │   └── test_fuse.py // script for testing the model
  ├── data_science_wordnet_en_0.2/ - folder for science wordnet
  │   ├── score_gnn.txt - scores for PGAT propogated embeddings
  │   ├── LD.txt, gene_diff.txt, nfd_norm.txt, LCS.txt, Contains.txt, Suffix.txt, Ends.txt  - value matrix of term pairs with 7 lexico-syntactic patterns 
  │   ├── paths.json - dependency path information for all possible paths
  │   ├── paths_index.json - the index information for all dependency paths
  │   ├── taxo_path.json - all the paths from the training set of the seed taxonomy
  │   ├── taxo_node_info.json - all the term information in the seed taxonomy
  ├── data_environment_eurovoc_en_0.2/ - folder for environment wordnet
  │   └── structure similar to above one
  └── log_results/ - store results
  ```

## Usage
Use `run.sh` from `model/` to run the code.
Some Key parameters:
```javascript
{
  "epochs": 20,             // number of training epochs
  "lr": 1e-3,               // number of learning rate
  "cudaid": 1,              // id of gpu
  "dropout": 0.4,           // dropout rate
  "hidden": 200,            // number of hidden layers
  "weight_decay": 5e-4,     // L2 Regularization
  "fp": "../data_environment_eurovoc_en_0.2",     // file path
  "path_len": 3,            // length of mini-path
  "lambda1": 0.1,           // weight of loss 1 (regularization for base classifiers)
  "lambda2": 0.1,           // weight of loss 2 (regularization for consistency)
  "taxi_feature": 1         // whether to load lexico-syntactic embeddings
  "load_gcn": 1             // whether to load gnn-propogated term embeddings
}
```

Add addional configurations if you need.

## Processing Text Data on Your Own
The way to obtain your own corpus is described as follows

- For GNN-propagated embeddings:
  - Use `model/bert_emb_extractor.py` to obtain the BERT Embeddings of terms.
  - Please follow the link of the paper [TaxoExpan](https://github.com/mickeystroller/TaxoExpan) to generate the GNN-propagated embeddings for terms. 
- For text corpus / contextual features: 
  - To build everything from scratch, first download corpora such as [Wikipedia](https://dumps.wikimedia.org/), [UMBC](https://ebiquity.umbc.edu/resource/html/id/351/UMBC-webbase-corpus), and [1 Billion Word Language Model Benchmark](http://www.statmt.org/lm-benchmark/).
  - To preprocess the corpus, generate a vocabulary file and use the scripts based on [LexNET](https://github.com/vered1986/LexNET). Please refer to the instructions [here](https://github.com/vered1986/LexNET/wiki/Detailed-Guide). It may take several hours to finish this process. 
- For Lexico-Syntactic Features:
  - Use `model/gen_lexico_features.py` to generate linguistic patterns.
  - For term frequency patterns, please refer to the instructions [here](https://github.com/uhh-lt/taxi).
## TODOs

- [ ] Support more tensorboard functions
- [ ] Using fixed random seed


## Acknowledgements
If you find this paper useful for your research, please cite the following paper in your publication:

```
@inproceedings{yu2020steam,
  title={STEAM: Self-Supervised Taxonomy Expansion with Mini-Paths},
  author={Yu, Yue and Li, Yinghao and Shen, Jiaming and Feng, Hao and Sun, Jimeng and Zhang, Chao},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
  publisher = {ACM},
  year={2020}
}
```

