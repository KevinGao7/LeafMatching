# Complete-Tree Space Favors Data-Efficient Link Prediction

This repository provides the official implementation for the paper:

> **Complete-Tree Space Favors Data-Efficient Link Prediction**

## Repository Structure

The codebase is organized as follows:

```
./ 
├── dataset/ 
├── logs/ 
├── special_req/ 
│ └── embedding_link_prediction_dw.py 
├── main.py 
├── utils.py 
├── split_datasets.py
├── synthetic_datasets.py
├── dataset_data_efficiency.sh 
├── dataset_practicality.sh
├── dataset_scalability.sh
├── exp_data_efficiency.sh
├── exp_practicality.sh 
└── exp_scalability.sh 
```

- **dataset/**: Where the datasets (like `Cora`) should be placed.  
- **logs/**: Output directory for training and testing logs.  
- **special_req/**: A modified file for CogDL’s `embedding_link_prediction_dw.py`.  
- **main.py**: Entry point for main experiments.  
- **utils.py**: Utility functions.  
- **split_datasets.py**: Dataset splitter in our setting.
- **bash scripts**: Shell scripts to run different experiment scenarios.

---

## Requirements

1. **Python Environment**  
   Make sure you have the following packages installed:
   
   - `pytorch`
   - `scikit-learn`
   - `numpy`
   - `pandas`
   - `networkx`
   - `torch_geometric`
   
2. **Datasets**  
   We use [CogDL](https://github.com/THUDM/cogdl) for automatic dataset downloading. Please install CogDL in your environment
   
   ```bash
   pip install cogdl
   ```
   or refer to CogDL's GitHub repo for detailed installation instructions.
   
3. **Replace the CogDL Link Prediction Wrapper**
   In your local CogDL installation, find the file:

   ```
   cogdl/wrappers/data_wrapper/link_prediction/embedding_link_prediction_dw.py
   ```

   Replace its content with the file provided in our repository:

   ```
   ./special_req/embedding_link_prediction_dw.py
   ```

   This step ensures compatibility with our experimental setup.

## Reproducing Experiments

We provide several pre-configured bash scripts to reproduce different experiment settings described in the paper. All results will be logged in `./logs/` for further analysis. As default, all the metrics including `roc_auc, pr_auc, mrr, f1, hits20, hits50, hits100` would be report in every epoch for **performance comparison**. The loss of each epoch is reported for **convergence judgement**. 

When splitting data, `CogDL` or `torch_geometric` may fail to download the dataset. Please manually download them in `./datasets`.

### Data Efficiency

To reproduce the data-efficiency experiments (varying $\mu$ from 0.1 to 0.9 on `cora`, `citeseer`, `pubmed`, `icews18` and `ogbl-collab`):

```bash
bash ./dataset_data_efficiency.sh
bash ./exp_data_efficiency.sh
```

###  Practicality

To assess the model’s performance on `ogbl-collab` and `ogbl-ppa` with $\mu = 0.02$:

```bash
bash ./data_practicality.sh
bash ./exp_practicality.sh
```

###  Scalability

To evaluate scalability on synthetic graphs and the real graph (ogbl-collab):

```bash
bash ./dataset_scalability.sh
bash ./exp_scalability.sh
```
