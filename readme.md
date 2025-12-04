# CrossMiT: Cross-Domain Transfer Framework for Enhanced miRNA–Target Interaction Prediction via Joint Learning

CrossMiT predicts miRNA-Target Interactions by leveraging miRNA-Disease Associations to overcome data sparsity. It employs a bi-directional graph transfer network to align genotypic and phenotypic features through shared miRNA anchors.

![](https://github.com/MT-Chuyen/CrossMiT/raw/main/Flow.jpg)

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

The execution process consists of 2 main steps:

### Step 1: Prepare Data

Run the `Prepare_data.py` script to process the raw data and create the 5-Fold structure.

```bash
cd Code
python Prepare_data.py
```

After this script finishes, the `Data/Data-kFold/` directory will be created, containing the pre-split data for all 5 folds.

### Step 2: Train and Evaluate (5-Fold Cross-Validation)

Run the `Run_all.py` script to automatically perform the training and evaluation process across all 5 folds. This script will call `Main.py` for each fold and summarize the results.

```bash
# Still inside the Code directory
python Run_all.py
```

The final aggregated results will be saved in `Data/results_summary.txt`.

### (Optional) Run Manually on a Single Fold

If you only want to run on a specific fold (e.g., Fold 1), you can run `Main.py` directly and specify the desired fold.

```bash
cd Code
python Main.py --fold 1 --gpu_id 0
```
 
 
