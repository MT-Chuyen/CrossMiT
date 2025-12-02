# CrossMiT: Cross-Domain Transfer Framework for Enhanced miRNA–Target Interaction Prediction via Joint Learning

CrossMiT predicts miRNA-Target Interactions by leveraging miRNA-Disease Associations to overcome data sparsity. It employs a bi-directional graph transfer network to align genotypic and phenotypic features through shared miRNA anchors.

![](https://github.com/MT-Chuyen/CrossMiT/raw/main/workflowx.png)

## 📂 Repo Structure  

* **`Data/`**: Contains the data used and the data processing files.
* **`Code/`**: Contains all source code to reproduce all the results
 


## Description Details
* **`Code/`**:
- Main.py: Main training loop. Load data, initialize CrossMiT model, save checkpoint, handle resume/pretrain.
- Model.py: Defines the CrossMiT model architecture.
- Utility.py: Utility functions: Calculating metrics: getHitRatio, getNDCG; Test support: get_test_instance; Directory management: ensureDir; Early stopping: early_stopping; Log printing: pprint
Flow: Main.py runs training → uses Model.py to build the model → uses Utility.py to calculate metrics and manage files.

* **`Code/`**:
- split_data.py: Split train/test data (split_data, split_loo for leave-one-out).

- CSV files: Rating data (miRNA-disease, miRNA-gene).

- miRNA-disease_miRNA-target/, miRNA-target_miRNA-disease/ directories: Contains adjacency matrix (.npz) and processed data for the model.
Flow: Main.py runs training → uses Model.py to build the model → uses Utility.py to calculate metrics and manage files.

Comman to run code: python Main.py

---

## 🚀 How to Run  
 

* Download the repo

* Follow instructions in the folder Code to run
 
