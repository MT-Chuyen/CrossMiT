# CrossMiT: Cross-Domain Transfer Framework for Enhanced miRNA–Target Interaction Prediction via Joint Learning

CrossMiT predicts miRNA-Target Interactions by leveraging miRNA-Disease Associations to overcome data sparsity. It employs a bi-directional graph transfer network to align genotypic and phenotypic features through shared miRNA anchors.

![](https://github.com/MT-Chuyen/CrossMiT/raw/main/Flow.jpg)

## 📂 Repo Structure  

*   **`Code/Prepare_data.py`**: The initial script. It reads data from `miRNA-target.csv` and `miRNA-disease.csv`, finds common miRNAs, re-indexes them, and creates a 5-fold cross-validation data structure in `Data/Data-kFold/`.
*   **`Code/Run_all.py`**: A script to automate the entire experiment. It sequentially calls `Main.py` to train and evaluate on each fold, then aggregates the final results.
*   **`Code/Main.py`**: The "heart" of the project, responsible for training and evaluating the model on a specific data fold. It initializes the model, manages the training loop, calls the `test` function, and saves checkpoints and logs.
*   **`Code/Model.py`**: Defines the neural network architecture of CrossMiT, including Graph Convolutional Layers and the knowledge transfer mechanism between the two domains.
*   **`Code/Utility.py`**: Contains crucial support components:
    *   `Data` class: Loads and manages training/testing data for each fold.
    *   `test` function: Performs the evaluation process and calculates metrics (HR, NDCG, AUC, MAP, etc.).
    *   Other utility functions like `early_stopping` and `pprint`.


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
 
 
