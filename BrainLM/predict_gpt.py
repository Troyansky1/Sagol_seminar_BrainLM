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
moving_window_len = 40  # must be divisible by the model's `timepoint_patching_size`

# ---- Paths ----
checkpoint_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrainLM/BrainLM/training-runs/pretrain_2025-05-12-12_49_38_/checkpoint-2900"
test_ds_path = "/home/ai_center/ai_data/gonyrosenman/postprocess_results/brain_LM_regular/test"  # arrow files?
coords_ds_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrainLM/BrainLM/toolkit/atlases/A424_Coordinates.dat"

# ---- Load model ----
model = BrainLMForPretraining.from_pretrained(checkpoint_path)
model.eval()
print("Loaded model from", checkpoint_path)

# ---- Load dataset ----
test_ds = load_from_disk(test_ds_path)
print("Loaded test dataset with", len(test_ds), "examples")

# ---- Load coordinates ----
coords_df = pd.read_csv(coords_ds_path, delimiter="\t", names=["idx", "X", "Y", "Z"])
coords_ds = Dataset.from_pandas(coords_df)
print("Loaded", len(coords_ds), "parcel coordinates")


# ---- Pre-processing helpers ----

def preprocess_batch(examples):
    """Prepare a mini-batch that matches the BrainLM input interface.


    Returned keys: `signal_vectors`, `xyz_vectors`, and `padding_mask`, each
    a **torch.Tensor**.
    """

    sig_list, xyz_list, pad_list = [], [], []
    batch_size = len(examples[recording_col_name])

    for signal_np in examples[recording_col_name]:
        # Convert to tensor  [T, V]
        signal_tensor = torch.tensor(signal_np, dtype=torch.float32)

        # Choose deterministic first window (for reproducibility) – change to randint for stochastic windows
        start_idx = 0  # randint(0, signal_tensor.shape[0] - moving_window_len)
        window = signal_tensor[start_idx : start_idx + moving_window_len]  # [T, V]

        # Rearrange to [V, T]
        window = window.T.contiguous()
        num_voxels = window.shape[0]

        sig_list.append(window)

        # Spatial coords  [V, 3]
        xyz_tensor = torch.stack(
            [
                torch.tensor([
                    coords_ds[i]["X"],
                    coords_ds[i]["Y"],
                    coords_ds[i]["Z"],
                ], dtype=torch.float32)
                for i in range(num_voxels)
            ],
            dim=0,
        )
        xyz_list.append(xyz_tensor)

        # Padding mask – all ones ("valid") because we have no padding time-steps.
        # Shape must mirror signal_vectors: [V, T]
        pad_list.append(torch.ones_like(window, dtype=torch.bool))

    batch_signal = torch.stack(sig_list, dim=0)  # [B, V, T]
    batch_xyz = torch.stack(xyz_list, dim=0)     # [B, V, 3]
    batch_pad = torch.stack(pad_list, dim=0)     # [B, V, T]

    return {
        "signal_vectors": batch_signal,
        "xyz_vectors": batch_xyz,
        "padding_mask": batch_pad,
    }


# ---- Prepare a single example (batch size = 1) ----
example = {recording_col_name: [test_ds[0][recording_col_name]]}
inputs = preprocess_batch(example)
print("Input tensor shapes → signals:", inputs["signal_vectors"].shape, ", xyz:", inputs["xyz_vectors"].shape)

# BrainLM does not use `labels` during inference, but the signature requires it.
labels_stub = torch.zeros(1, dtype=torch.float32)

# ---- Inference ----
with torch.no_grad():
    output = model(
        signal_vectors=inputs["signal_vectors"],
        xyz_vectors=inputs["xyz_vectors"],
        labels=labels_stub,
        input_ids=inputs["signal_vectors"],  # unused – kept for API compatibility
        padding_mask=inputs["padding_mask"],
    )

# ---- Post-processing ----
logits_tensor = output.logits[0]  # (logits, latent) → take first element
logits = logits_tensor[0]         # remove batch dim → [V, num_tokens, patch_size]
logits_flat = logits.view(logits.shape[0], -1)  # [V, T]

mask = output.mask[0].cpu().numpy()  # [V, T]
gt = inputs["signal_vectors"][0].cpu().numpy()
pred = logits_flat.cpu().numpy()

print("Prediction tensor shape:", logits_flat.shape)
print("Ground-truth shape:   ", gt.shape)

# ---- Plot helper ----
os.makedirs("plots", exist_ok=True)

def plot_parcel(idx: int):
    """Save GT vs. prediction curve for a single parcel."""
    prediction = pred[idx]
    masked_prediction = np.where(mask[idx], prediction, np.nan)

    plt.figure()
    plt.plot(gt[idx], label="Ground Truth")
    plt.plot(masked_prediction, label="Prediction (masked only)")

    for t in range(mask.shape[1]):
        if mask[idx, t]:
            plt.axvline(x=t, color='gray', alpha=0.2, linestyle='--')

    plt.title(f"Parcel {idx}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/parcel_{idx}.png")
    plt.close()


# Example visualisations
plot_parcel(1)
plot_parcel(2)
print("Saved plots to ./plots")