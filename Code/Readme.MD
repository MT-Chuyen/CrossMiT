
# Code Descriptions
- Main.py: Main training loop. Load data, initialize CrossMiT model, save checkpoint, handle resume/pretrain.
- Model.py: Defines the CrossMiT model architecture.
- Utility.py: Utility functions: Calculating metrics: getHitRatio, getNDCG; Test support: get_test_instance; Directory management: ensureDir; Early stopping: early_stopping; Log printing: pprint

Flow: Main.py runs training → uses Model.py to build the model → uses Utility.py to calculate metrics and manage files.

Comman to run code: python Main.py
