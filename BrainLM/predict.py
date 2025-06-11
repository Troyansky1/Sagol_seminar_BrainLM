
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
        # start_idx = randint(0, signal_vector.shape[0] - moving_window_len)
        start_idx = 0
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
print("inputs shape", inputs["signal_vectors"].shape)
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
logits_tensor = output["logits"][0]  # [1, 424, 3, 40]
# Concatenate the 3 tokens per voxel
logits_reshaped = logits_tensor[0].reshape(424, -1)  # [424, 120]

# Get full mask pattern for all tokens
mask_full = output["mask"][0].reshape(424, 3)  # [424 voxels, 3 tokens each]

# Find voxels with exactly 1 masked token
num_masked_per_voxel = torch.sum(mask_full, dim=1)  # Count masked tokens per voxel
single_masked_voxels = torch.where(num_masked_per_voxel == 1)[0]
#logits = output["logits"][0, :, :, 0].cpu().numpy()  # shape: [voxels, tokens]
gt = inputs["signal_vectors"][0, :, :].cpu().numpy()

# ---- Plot ----
os.makedirs("plots", exist_ok=True)

def plot_single_masked_voxel(voxel_idx, gt, predicted_logits, mask_pattern):
    plt.figure(figsize=(15, 8))
    
    # Plot ground truth
    plt.plot(gt[voxel_idx, :], 'b-', linewidth=3, label="Ground Truth", alpha=0.8)
    
    # Plot prediction
    plt.plot(predicted_logits[voxel_idx, :], 'r--', linewidth=2, label="Prediction")
    
    # Highlight which 40-timepoint segment was masked
    masked_token_idx = torch.where(mask_pattern == 1)[0][0].item()  # Find which token was masked
    start_t = masked_token_idx * 40
    end_t = (masked_token_idx + 1) * 40
    
    plt.axvspan(start_t, end_t, alpha=0.3, color='red', 
               label=f'Masked Segment ({start_t}-{end_t})')
    
    # Add vertical lines to show token boundaries
    for i in range(1, 3):
        plt.axvline(x=i*40, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Voxel {voxel_idx} - Single Masked Token (Token {masked_token_idx})')
    plt.xlabel('Timepoint')
    plt.ylabel('Brain Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
        # Add text annotations for token regions
    plt.text(20, plt.ylim()[1]*0.9, 'Token 0\n(0-39)', ha='center', va='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if masked_token_idx != 0 else 'lightcoral'))
    plt.text(60, plt.ylim()[1]*0.9, 'Token 1\n(40-79)', ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if masked_token_idx != 1 else 'lightcoral'))
    plt.text(100, plt.ylim()[1]*0.9, 'Token 2\n(80-119)', ha='center', va='top',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen' if masked_token_idx != 2 else 'lightcoral'))
    
    plt.savefig(f"plots/single_masked_voxel_{voxel_idx}_token_{masked_token_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()


# Apply linear correction to match input statistics
def correct_predictions(pred, target_mean, target_std, pred_mean, pred_std):
    # Standardize predictions, then rescale to target distribution
    pred_standardized = (pred - pred_mean) / pred_std
    pred_corrected = pred_standardized * target_std + target_mean
    return pred_corrected

# Get statistics
input_mean = inputs['signal_vectors'].mean().item()
input_std = inputs['signal_vectors'].std().item()
output_mean = logits_reshaped.mean().item()
output_std = logits_reshaped.std().item()

# Apply correction
logits_corrected = correct_predictions(
    logits_reshaped.cpu().numpy(), 
    input_mean, input_std, 
    output_mean, output_std
)

print("=== AFTER CORRECTION ===")
print(f"Corrected mean: {logits_corrected.mean():.3f}")
print(f"Corrected std: {logits_corrected.std():.3f}")
print(f"Corrected min: {logits_corrected.min():.3f}")
print(f"Corrected max: {logits_corrected.max():.3f}")

# Test on unmasked voxel
no_masked_voxels = torch.where(num_masked_per_voxel == 0)[0]
if len(no_masked_voxels) > 0:
    test_voxel = no_masked_voxels[0].item()
    gt_signal = gt[test_voxel, :]
    pred_signal = logits_reshaped.cpu().numpy()[test_voxel, :]

test_voxel = no_masked_voxels[0].item()
gt_signal = gt[test_voxel, :]
pred_corrected_signal = logits_corrected[test_voxel, :]

mse = np.mean((gt_signal - pred_signal) ** 2)
mse_corrected = np.mean((gt_signal - pred_corrected_signal) ** 2)
print(f"MSE after correction: {mse_corrected:.6f} (was {mse:.6f})")

# Plot with corrected predictions
def plot_corrected_voxel(voxel_idx, gt, predicted_logits, mask_pattern):
    plt.figure(figsize=(15, 8))
    
    plt.plot(gt[voxel_idx, :], 'b-', linewidth=3, label="Ground Truth", alpha=0.8)
    plt.plot(predicted_logits[voxel_idx, :], 'r--', linewidth=2, label="Corrected Prediction")
    
    masked_token_idx = torch.where(mask_pattern == 1)[0][0].item()
    start_t = masked_token_idx * 40
    end_t = (masked_token_idx + 1) * 40
    
    plt.axvspan(start_t, end_t, alpha=0.3, color='red', 
               label=f'Masked Segment ({start_t}-{end_t})')
    
    for i in range(1, 3):
        plt.axvline(x=i*40, color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Voxel {voxel_idx} - Corrected Prediction (Token {masked_token_idx})')
    plt.xlabel('Timepoint')
    plt.ylabel('Brain Activity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f"plots/corrected_voxel_{voxel_idx}_token_{masked_token_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Plot one corrected example
if len(single_masked_voxels) > 0:
    voxel_idx = single_masked_voxels[0].item()
    mask_pattern = mask_full[voxel_idx]
    plot_corrected_voxel(voxel_idx, gt, logits_corrected, mask_pattern)