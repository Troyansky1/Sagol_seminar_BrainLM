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



def collate_fn(examples):
    """
    This function tells the dataloader how to stack a batch of examples from the dataset.
    Need to stack gene expression vectors and maintain same argument names for model inputs
    which CellLM is expecting in forward() function:
        expression_vectors, sampled_gene_indices, and cell_indices
    """
    labels = torch.zeros(len(examples["xyz_vectors"]), dtype=torch.float32)

    # These inputs will go to model.forward(), names must match
    return {
        "signal_vectors": examples["signal_vectors"][0].unsqueeze(0),
        "xyz_vectors": examples["xyz_vectors"][0].unsqueeze(0),
        "input_ids": examples["signal_vectors"][0].unsqueeze(0),
        "labels": labels
    }


def preprocess_images(examples):
    """
    Preprocessing function for dataset samples. This function is passed into Trainer as
    a preprocessor which takes in one row of the loaded dataset and constructs a model
    input sample according to the arguments which model.forward() expects.

    The reason this function is defined inside on main() function is because we need
    access to arguments such as cell_expression_vector_col_name.
    """
    #
    signal_val_list = []
    xyz_list = []
    # Get batch size
    batch_size = len(examples[recording_col_name])
    for idx, signal_vector in enumerate(examples[recording_col_name]):
        # signal_vector is list of lists of shape [num_timepoints x 424]
        signal_vector = torch.tensor(signal_vector, dtype=torch.float32)

        # Choose random starting index, take window of moving_window_len points for each region
        start_idx = randint(0, signal_vector.shape[0] - moving_window_len)
        end_idx = start_idx + moving_window_len

        # Append signal values and coords
        window = signal_vector[start_idx: end_idx, :]  # [moving_window_len, 424]
        num_brain_parcels = window.shape[1]
        for brain_region_idx in range(num_brain_parcels): 
            for idx, timepoint_idx in enumerate(range(start_idx, end_idx)): #iterate over timepoints (window size)
                xyz = torch.tensor([
                    coords_ds[brain_region_idx]["X"],
                    coords_ds[brain_region_idx]["Y"],
                    coords_ds[brain_region_idx]["Z"]], dtype=torch.float32)
                #xyzt = torch.cat([xyz, torch.tensor([timepoint_idx], dtype=torch.float32)])
                xyz_list.append(xyz)

                signal_val = window[idx, brain_region_idx]
                signal_val_list.append(torch.tensor([signal_val], dtype=torch.float32))
                
    print("xyz_vectors:", torch.stack(xyz_list, dim=0).shape)
    # Add in key-value pairs for model inputs which CellLM is expecting in forward() function:
    #  signal_vectors and xyz_vectors
    #  These lists will be stacked into torch Tensors by collate() function (defined above).
    examples["signal_vectors"] = torch.stack(signal_val_list, dim=0).reshape(batch_size, num_brain_parcels, -1)

    examples["xyz_vectors"] = torch.stack(xyz_list, dim=0).reshape(batch_size, -1, 3)
    return examples

if not os.path.exists("plots"): #TODO change
    os.mkdir("plots")  # Creates folder called plots in root of project for saving plots

checkpoint_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/training-runs/pretrain_2025-05-12-12_49_38_/checkpoint-2900"

model = BrainLMForPretraining.from_pretrained(checkpoint_path)
print("model")

model.config


test_ds_path = "/home/ai_center/ai_data/gonyrosenman/postprocess_results/brain_LM_regular/test" #arrow files?
test_ds = load_from_disk(test_ds_path)
print("test_ds")

example1 = test_ds[0]
print(np.array(example1["Voxelwise_RobustScaler_Normalized_Recording"]).shape)


coords_ds_path = "/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrinLM/BrainLM/toolkit/atlases/A424_Coordinates.dat"
if '.dat' in str(coords_ds_path):
    coords_ds = pd.read_csv('/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/Sagol_seminar_BrinLM/BrainLM/toolkit/atlases/A424_Coordinates.dat',delimiter='\t', names=['idx', 'X', 'Y', 'Z'])
    coords_ds = Dataset.from_pandas(coords_ds)
else:
    coords_ds = load_from_disk(coords_ds_path)
print("coords_ds")

recording_col_name = "Voxelwise_RobustScaler_Normalized_Recording"
moving_window_len = 120 #TODO ask gony


example = test_ds[0] #TODO: on entire dataset?

# Wrap each value in the key:value pairs into a list (expected by preprocess() and collate())
example["Voxelwise_RobustScaler_Normalized_Recording"] = [example["Voxelwise_RobustScaler_Normalized_Recording"]]

processed_example = preprocess_images(
    examples=example

)  # preprocess samples genes, adds keys, etc
model_inputs = collate_fn(processed_example)
print(model_inputs)

print(model_inputs["signal_vectors"].shape)
print(model_inputs["xyz_vectors"].shape)

model.vit.embeddings.mask_for_inference = True

vitmae_for_pre_training_output = model(
    signal_vectors=model_inputs["signal_vectors"],
    xyz_vectors=model_inputs["xyz_vectors"],
    labels=model_inputs["labels"],
    input_ids=model_inputs["input_ids"]
)
print(vitmae_for_pre_training_output.keys())

vitmae_for_pre_training_output["loss"]

vitmae_for_pre_training_output["logits"].shape


print(vitmae_for_pre_training_output["mask"].shape)
print(vitmae_for_pre_training_output["mask"][0, :12])


preds = vitmae_for_pre_training_output["logits"][0, :, 0].detach().cpu().numpy()
print(preds.shape)
gt = model_inputs["signal_vectors"][0, :, 0].detach().cpu().numpy()
print(gt.shape)


plt.plot(gt[:5], label="Input Data")
plt.plot(preds[5:10], label="Prediction")
plt.savefig("plot1.png")


plt.plot(gt[110:115], label="Input Data")
plt.plot(preds[115:120], label="Prediction")
plt.savefig("plot2.png")