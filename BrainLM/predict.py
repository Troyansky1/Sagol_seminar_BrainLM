
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

"""
Simple test to verify ids_restore logic works correctly with synthetic data.
Add this to your train.py right after model creation, before any training.
"""

def test_ids_restore_logic():
    """
    Test the ids_restore mechanism with simple synthetic data.
    This mimics what happens in BrainLMEmbeddings.random_masking()
    """
    print("=" * 50)
    print("TESTING IDS_RESTORE LOGIC WITH SYNTHETIC DATA")
    print("=" * 50)
    
    # Create simple synthetic data
    batch_size = 2
    seq_length = 10  # Small sequence for easy visualization
    hidden_dim = 4
    mask_ratio = 0.4  # Mask 40%
    
    device = next(model.parameters()).device
    
    # Create test sequence where each position has its index as value
    # This makes it easy to see if restoration works
    test_sequence = torch.zeros(batch_size, seq_length, hidden_dim, device=device)
    for i in range(seq_length):
        test_sequence[:, i, :] = i  # Position 0 has value 0, position 1 has value 1, etc.
    
    print("Original sequence (each position has its index as value):")
    print("Shape:", test_sequence.shape)
    print("Batch 0:")
    for i in range(seq_length):
        print(f"  Position {i}: {test_sequence[0, i, 0].item()}")
    
    # Step 1: Create random shuffle (like in random_masking)
    noise = torch.rand(batch_size, seq_length, device=device)
    ids_shuffle = torch.argsort(noise, dim=1)
    ids_restore = torch.argsort(ids_shuffle, dim=1)
    
    print(f"\nRandom shuffle order: {ids_shuffle[0].tolist()}")
    print(f"Restore order: {ids_restore[0].tolist()}")
    
    # Step 2: Keep only unmasked tokens (like encoder does)
    len_keep = int(seq_length * (1 - mask_ratio))
    ids_keep = ids_shuffle[:, :len_keep]
    
    print(f"\nKeeping first {len_keep} positions after shuffle: {ids_keep[0].tolist()}")
    
    # Extract only the kept tokens
    sequence_unmasked = torch.gather(
        test_sequence, 
        dim=1, 
        index=ids_keep.unsqueeze(-1).repeat(1, 1, hidden_dim)
    )
    
    print(f"\nUnmasked sequence (what encoder sees):")
    print("Shape:", sequence_unmasked.shape)
    for i in range(len_keep):
        original_pos = ids_keep[0, i].item()
        print(f"  Kept token {i}: value={sequence_unmasked[0, i, 0].item()} (from original position {original_pos})")
    
    # Step 3: Simulate decoder - add mask tokens and restore order
    num_mask_tokens = seq_length - len_keep
    mask_tokens = torch.full((batch_size, num_mask_tokens, hidden_dim), -999.0, device=device)  # Use -999 to clearly see mask tokens
    
    print(f"\nAdding {num_mask_tokens} mask tokens with value -999")
    
    # Concatenate unmasked + mask tokens
    full_sequence = torch.cat([sequence_unmasked, mask_tokens], dim=1)
    
    print(f"Full sequence before restore (unmasked + mask tokens):")
    for i in range(seq_length):
        print(f"  Position {i}: {full_sequence[0, i, 0].item()}")
    
    # Step 4: Restore original order using ids_restore
    restored_sequence = torch.gather(
        full_sequence,
        dim=1,
        index=ids_restore.unsqueeze(-1).repeat(1, 1, hidden_dim)
    )
    
    print(f"\nRestored sequence (after applying ids_restore):")
    for i in range(seq_length):
        print(f"  Position {i}: {restored_sequence[0, i, 0].item()}")
    
    # Step 5: Create the mask to see which positions were masked
    mask = torch.ones(batch_size, seq_length, device=device)
    mask[:, :len_keep] = 0  # 0 = unmasked, 1 = masked
    mask = torch.gather(mask, dim=1, index=ids_restore)
    
    print(f"\nFinal mask (0=unmasked, 1=masked): {mask[0].tolist()}")
    
    # Step 6: Verify correctness
    print("\n" + "=" * 30)
    print("VERIFICATION:")
    print("=" * 30)
    
    # Check if unmasked positions have correct values
    unmasked_positions = (mask[0] == 0).nonzero().flatten()
    all_correct = True
    
    for pos in unmasked_positions:
        pos = pos.item()
        expected_value = pos  # Should equal position index
        actual_value = restored_sequence[0, pos, 0].item()
        is_correct = (actual_value == expected_value)
        print(f"Position {pos}: expected {expected_value}, got {actual_value} - {'✓' if is_correct else '✗'}")
        if not is_correct:
            all_correct = False
    
    # Check if masked positions have mask token value
    masked_positions = (mask[0] == 1).nonzero().flatten()
    for pos in masked_positions:
        pos = pos.item()
        actual_value = restored_sequence[0, pos, 0].item()
        is_mask_token = (actual_value == -999.0)
        print(f"Position {pos}: masked, got {actual_value} - {'✓' if is_mask_token else '✗'}")
        if not is_mask_token:
            all_correct = False
    
    print(f"\nOVERALL RESULT: {'✅ PASS' if all_correct else '❌ FAIL'}")
    
    if all_correct:
        print("ids_restore logic is working correctly!")
    else:
        print("❌ BUG DETECTED in ids_restore logic!")
        print("This explains why your model can't reconstruct properly.")
    
    return all_correct

# Run the test
test_result = test_ids_restore_logic()

# Test with your actual model's masking function
def test_with_actual_model():
    """
    Test using your actual model's embedding layer
    """
    print("\n" + "=" * 50)
    print("TESTING WITH ACTUAL MODEL EMBEDDING LAYER")
    print("=" * 50)
    
    # Create synthetic brain data that matches your model's expected input
    batch_size = 1
    num_voxels = 424  # Your brain atlas size
    num_timepoints = 200  # Your window size
    
    device = next(model.parameters()).device
    
    # Create synthetic signal where each voxel has a different constant value
    synthetic_signals = torch.zeros(batch_size, num_voxels, num_timepoints, device=device)
    for v in range(num_voxels):
        synthetic_signals[0, v, :] = v  # Voxel 0 has value 0, voxel 1 has value 1, etc.
    
    # Create dummy xyz coordinates
    synthetic_xyz = torch.randn(batch_size, num_voxels, 3, device=device)
    
    print(f"Created synthetic data:")
    print(f"  Signals shape: {synthetic_signals.shape}")
    print(f"  XYZ shape: {synthetic_xyz.shape}")
    print(f"  Each voxel has constant value equal to its index")
    
    # Run through your model's embedding layer
    model.eval()
    with torch.no_grad():
        embeddings, mask, ids_restore = model.vit.embeddings(
            synthetic_signals, 
            synthetic_xyz, 
            noise=None
        )
    
    print(f"\nModel embedding results:")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Mask shape: {mask.shape}")
    print(f"  ids_restore shape: {ids_restore.shape}")
    print(f"  Mask ratio: {mask.mean().item():.3f}")
    
    # Now test reconstruction through decoder
    decoder_output = model.decoder(embeddings, synthetic_xyz, ids_restore)
    reconstructed = decoder_output.logits
    
    print(f"\nDecoder output:")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    
    # Reshape original signals to match reconstructed format
    original_reshaped = torch.reshape(synthetic_signals, reconstructed.shape)
    
    # Check if unmasked regions are reconstructed correctly
    mask_reshaped = mask.reshape(reconstructed.shape[:-1])
    unmasked_mask = (mask_reshaped == 0)
    
    if unmasked_mask.any():
        # Compare unmasked regions
        original_unmasked = original_reshaped[unmasked_mask.unsqueeze(-1).repeat(1, 1, 1, reconstructed.shape[-1])]
        reconstructed_unmasked = reconstructed[unmasked_mask.unsqueeze(-1).repeat(1, 1, 1, reconstructed.shape[-1])]
        
        mse = ((original_unmasked - reconstructed_unmasked) ** 2).mean()
        print(f"\nUnmasked region MSE: {mse.item():.6f}")
        
        if mse.item() < 0.01:
            print("✅ Unmasked regions reconstructed well!")
        else:
            print("❌ Unmasked regions NOT reconstructed well - ids_restore or decoder issue!")
    
    return mse.item() if unmasked_mask.any() else float('inf')

# Run both tests
if test_result:
    actual_model_mse = test_with_actual_model()
else:
    print("Skipping actual model test due to basic logic failure")

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
no_masked_voxels = torch.where(num_masked_per_voxel == 0)[0]

gt = inputs["signal_vectors"][0, :, :].cpu().numpy()  # [424, 120]

# ---- Plot ----
os.makedirs("plots", exist_ok=True)

def plot_single_masked_voxel(voxel_idx, gt, predicted_logits, mask_pattern):
    plt.figure(figsize=(15, 8))
    
    # Plot ground truth
    plt.plot(gt[voxel_idx, :], 'b-', linewidth=3, label="Ground Truth", alpha=0.8)
    
    # Plot corrected prediction
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
    
    plt.title(f'Voxel {voxel_idx} - Prediction (Token {masked_token_idx})')
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
    
    plt.savefig(f"plots/parcel_{voxel_idx}_token_{masked_token_idx}.png", dpi=150, bbox_inches='tight')
    plt.close()

# Plot first 3 voxels with exactly 1 masked token using  predictions
if len(single_masked_voxels) > 0:
    print(f"Plotting {min(3, len(single_masked_voxels))} single-masked voxels...")
    for i in range(min(3, len(single_masked_voxels))):
        voxel_idx = single_masked_voxels[i].item()
        mask_pattern = mask_full[voxel_idx]
        plot_single_masked_voxel(voxel_idx, gt, logits_reshaped, mask_pattern)
else:
    print("No voxels found with exactly 1 masked token!")

