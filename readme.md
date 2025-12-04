# CrossMiT: Cross-Domain Transfer Framework for Enhanced miRNA–Target Interaction Prediction via Joint Learning

CrossMiT predicts miRNA-Target Interactions by leveraging miRNA-Disease Associations to overcome data sparsity. It employs a bi-directional graph transfer network to align genotypic and phenotypic features through shared miRNA anchors.

![CrossMiT Framework](https://github.com/MT-Chuyen/CrossMiT/raw/main/Flow.jpg)


## 📂 Repo Structure  
The project is organized into two main directories: `Data` for data storage and `Code` for the source code.
```
CrossMiT/
├── Code/
│   ├── Prepare_data.py     # STEP 1: Preprocess and split data into K-Folds.
│   ├── Run_all.py          # STEP 2: Automatically run 5-Fold CV and summarize results.
│   ├── Main.py             # Main script to train and evaluate on a single Fold.
│   ├── Model.py            # Defines the CrossMiT model architecture.
│   ├── Utility.py          # Contains the Data class, test function, and other helpers.
│   ├── Config.py           # Manages command-line arguments.
│  
│
└── Data/
    ├── miRNA-target.csv    # Raw data: miRNA–Target interactions.
    ├── miRNA-disease.csv   # Raw data: miRNA–Disease associations.
    │
    ├── Data-kFold/         # (Will be created) Directory for processed K-Fold data.
    │   └── Fold_1/
    │       ├── miRNA-disease_miRNA-target/ # Source domain data (disease).
    │       │   ├── train.txt
    │       │   ├── test.txt
    │       │   └── test_neg.txt
    │       └── miRNA-target_miRNA-disease/ # Target domain data (target).
    │           └── ...
    │   └── Fold_2/, Fold_3/, ...
    │
    ├── logs/               # (Will be created) Stores detailed logs for each run.
    ├── weights/            # (Will be created) Stores trained model weights.
    └── output/             # (Will be created) Stores evaluation results (metrics).
        ├── folds/
```
---

## 🚀 How to Run  

The execution process consists of 2 main steps:

### Step 1: Prepare Data

Run the `Prepare_data.py` script to process the raw data.

```bash
cd Code
python Prepare_data.py
```

After this script finishes, the `Data/Data-kFold/` directory will be created, containing the pre-split data for all 5 folds.

### Step 2: Train and Evaluate  

Run the `Run_all.py` script to automatically perform the training and evaluation process across all 5 folds. This script will call `Main.py` for each fold and summarize the results.

```bash
# Still inside the Code directory
python Run_all.py
```
 
 
 
