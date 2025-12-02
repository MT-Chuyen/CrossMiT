# CrossMiT: Cross-Domain Transfer Framework for Enhanced miRNA–Target Interaction Prediction via Joint Learning

CrossMiT predicts miRNA-Target Interactions by leveraging miRNA-Disease Associations to overcome data sparsity. It employs a bi-directional graph transfer network to align genotypic and phenotypic features through shared miRNA anchors.

![](https://github.com/MT-Chuyen/CrossMiT/raw/main/workflowx.png)

## 📂 Repo Structure  

* **`Code/`**: Contains all source code to reproduce the results  
  - **Main.py**: Main training loop — loads data, initializes CrossMiT model, saves checkpoints, handles resume/pretrain.  
  - **Model.py**: Defines the CrossMiT model architecture.  
  - **Utility.py**: Utility functions:  
    - Calculating metrics: `getHitRatio`, `getNDCG`  
    - Test support: `get_test_instance`  
    - Directory management: `ensureDir`  
    - Early stopping: `early_stopping`  
    - Log printing: `pprint`  
  - **Flow**: Main.py → Model.py → Utility.py  

* **`Data/`**: Contains raw data and processing scripts  
  - **split_data.py**: Handles train/test split (`split_data`, `split_loo` for leave-one-out).  
  - **CSV files**: miRNA–disease and miRNA–gene rating data.  
  - **miRNA-disease_miRNA-target/** and **miRNA-target_miRNA-disease/**:  
    - Adjacency matrices (.npz)  
    - Processed data used by the model  

Flow: Main.py runs training → uses Model.py to build the model → uses Utility.py to calculate metrics and manage files.

 
---

## 🚀 How to Run  

command line: python Main.py
 
 
