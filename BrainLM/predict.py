
import os
from random import randint

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_from_disk, Dataset

from brainlm_mae.modeling_brainlm import BrainLMForPretraining

# ---- Parameters ----
recording_col_name = "Voxelwise_RobustScaler_Normalized_Recording"
moving_window_len = 120

# ---- Paths ----
checkpoint_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrainLM/BrainLM/training-runs/pretrain_2025-05-12-12_49_38_/checkpoint-2900"
test_ds_path = "/home/ai_center/ai_data/gonyrosenman/postprocess_results/brain_LM_regular/test" #arrow files?
coords_ds_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrainLM/BrainLM/toolkit/atlases/A424_Coordinates.dat"
    
# ---- Load model ----
model = BrainLMForPretraining.from_pretrained(checkpoint_path)
model.eval()
print("Loaded model")

# ---- Load dataset ----
test_ds = load_from_disk(test_ds_path)
print("Loaded test dataset")

# ---- Load coordinates ----
coords_ds = pd.read_csv(coords_ds_path, delimiter="\t", names=["idx", "X", "Y", "Z"])
coords_ds = Dataset.from_pandas(coords_ds)
#coords = torch.tensor(coords_ds[["X", "Y", "Z"]].values, dtype=torch.float32)
print("Loaded coordinates:", coords_ds.shape)

# ---- Preprocessing ----
def preprocess_images(examples):
    """
    Preprocess a batch of examples to produce:
    - signal_vectors: [B, V, T]
    - xyz_vectors: [B, V, 3]
    where V = number of brain parcels, T = timepoints per node (window), B = batch size
    """
    batch_signal_list = []
    batch_xyz_list = []

    batch_size = len(examples[recording_col_name])
    for signal_vector in examples[recording_col_name]:
        # Convert input signal to tensor: shape [T, V]
        signal_vector = torch.tensor(signal_vector, dtype=torch.float32)

        # Select a random time window
        start_idx = randint(0, signal_vector.shape[0] - moving_window_len)
        end_idx = start_idx + moving_window_len

        window = signal_vector[start_idx:end_idx, :]  # shape: [moving_window_len, V]
        num_brain_parcels = window.shape[1]

        # Transpose to [V, T] so each region has its own signal over time
        signal_tensor = window.T  # [V, T]
        batch_signal_list.append(signal_tensor)

        # Create one xyz vector per region (not per timepoint!)
        xyz_list = []
        for brain_region_idx in range(num_brain_parcels):
            xyz = torch.tensor([
                coords_ds[brain_region_idx]["X"],
                coords_ds[brain_region_idx]["Y"],
                coords_ds[brain_region_idx]["Z"]
            ], dtype=torch.float32)
            xyz_list.append(xyz)
        xyz_tensor = torch.stack(xyz_list, dim=0)  # [V, 3]
        batch_xyz_list.append(xyz_tensor)

    # Final shapes:
    # signal_vectors: [B, V, T]
    # xyz_vectors:    [B, V, 3]
    examples["signal_vectors"] = torch.stack(batch_signal_list, dim=0)
    examples["xyz_vectors"] = torch.stack(batch_xyz_list, dim=0)

    return examples



# ---- Prepare example ----
example = test_ds[0]
example[recording_col_name] = [example[recording_col_name]]  # wrap list
inputs = preprocess_images(example)

labels = torch.zeros(len(inputs["xyz_vectors"]), dtype=torch.float32)

# These inputs will go to model.forward(), names must match
model_inputs= {
    "signal_vectors": inputs["signal_vectors"][0].unsqueeze(0),
    "xyz_vectors": inputs["xyz_vectors"][0].unsqueeze(0),
    "input_ids": inputs["signal_vectors"][0].unsqueeze(0),
    "labels": labels
}

# ---- Inference ----
with torch.no_grad():
    output = model(
        signal_vectors=model_inputs["signal_vectors"],
        xyz_vectors=model_inputs["xyz_vectors"],
        labels=model_inputs["labels"],
        input_ids=model_inputs["input_ids"],
    )

# ---- Results ----
logits_tensor = output["logits"][0]  # get the actual tensor
logits = logits_tensor[0, :, :, 0].cpu().numpy()  # now this will work


#logits = output["logits"][0, :, :, 0].cpu().numpy()  # shape: [voxels, tokens]
gt = inputs["signal_vectors"][0, :, :].cpu().numpy()

print("logits:", logits.shape)
print("gt:", gt.shape)

# ---- Plot ----
os.makedirs("plots", exist_ok=True)

def plot_parcel(num_parcel):
    plt.figure()
    plt.plot(gt[num_parcel, :], label="Ground Truth")
    plt.plot(logits[num_parcel, :], label="Prediction")
    plt.legend()
    plt.savefig(f"plots/plot_voxel_{num_parcel}.png")

for i in range(1, 10):
    plot_parcel(i)

