# Matches our changes that are in modeling brainLM copy

# Inbuilt Python libraries
import os
from random import randint

# Third party library imports
import umap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_from_disk, Dataset

# Imports for model from local definition
from brainlm_mae.modeling_brainlm import BrainLMForPretraining



if __name__ == "__main__":
    checkpoint_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/training-runs/pretrain_2025-05-12-12_49_38_/checkpoint-2900"
    model = BrainLMForPretraining.from_pretrained(checkpoint_path)
    model.config
    test_ds_path = "/home/ai_center/ai_data/gonyrosenman/postprocess_results/brain_LM_regular/test" #arrow files?
    test_ds = load_from_disk(test_ds_path)
    coords_ds_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/toolkit/atlases/A424_Coordinates.dat"
    if '.dat' in str(coords_ds_path):
        coords_ds = pd.read_csv('/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/toolkit/atlases/A424_Coordinates.dat',delimiter='\t', names=['idx', 'X', 'Y', 'Z'])
        coords_ds = Dataset.from_pandas(coords_ds)
    else:
        coords_ds = load_from_disk(coords_ds_path)
    recording_col_name = "Voxelwise_RobustScaler_Normalized_Recording"
    moving_window_len = 120 #TODO ask gony
    example = test_ds[0] #TODO: on entire dataset?
    # Wrap each value in the key:value pairs into a list (expected by preprocess() and collate())
    example["Voxelwise_RobustScaler_Normalized_Recording"] = [example["Voxelwise_RobustScaler_Normalized_Recording"]]
    processed_example = preprocess_images(
        examples=example

    )  # preprocess samples genes, adds keys, etc
    model_inputs = collate_fn(processed_example)