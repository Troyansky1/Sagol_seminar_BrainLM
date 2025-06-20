"""
Huggingface training script for BrainLM. Based on Huggingface script for pretraining Vit_MAE:
https://github.com/huggingface/transformers/blob/main/examples/pytorch/image-pretraining/run_mae.py

Image dataset specifics removed, will be replaced with fmri data processing functions and options.
"""

import logging
import os
import sys
import math
from dataclasses import dataclass, field
from typing import Optional
from random import randint

import torch
import wandb
from datasets import load_from_disk, DatasetDict

# Shira
from datasets import Dataset
import pandas as pd
import json


import transformers
from transformers import (
    HfArgumentParser,
    TrainingArguments,
)
#Shira from gony
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version, send_example_telemetry
from transformers.utils.versions import require_version

from brainlm_mae.modeling_brainlm import BrainLMForPretraining
from brainlm_mae.configuration_brainlm import BrainLMConfig
from utils.brainlm_trainer import BrainLMTrainer
#Shira from gony
from utils.metrics import MetricsCalculator


""" Pre-training a 🤗 ViT model as an MAE (masked autoencoder), as proposed in https://arxiv.org/abs/2111.06377."""
logger = logging.getLogger(__name__)

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.28.0.dev0")
require_version("datasets>=1.8.0", "To fix: conda env create -f environment.yml")


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    train_dataset_path: str = field(
        metadata={
            "help": "Path to saved train arrow dataset of cell x gene expression matrix."
        }
    )
    val_dataset_path: str = field(
        metadata={
            "help": "Path to saved val arrow dataset of cell x gene expression matrix."
        }
    )
    coords_dataset_path: str = field(
        metadata={"help": "Path to saved arrow dataset of brain region coordinates."}
    )
    recording_col_name: str = field(
        default="Voxelwise_RobustScaler_Normalized_Recording",
        metadata={"help": "Column in dataset which contains recording for each patient. Choose from:"
                          "All_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_All_Voxel_Normalized_Recording, "
                          "Per_Patient_Per_Voxel_Normalized_Recording, "
                          "Per_Voxel_All_Patient_Normalized_Recording, "
                          "Subtract_Mean_Normalized_Recording, "
                          "or Subtract_Mean_Divide_Global_STD_Normalized_Recording"
                  }
    )
    variable_of_interest_col_name: str = field(
        default="Age.At.MHQ",
        metadata={
            "help": "Column in dataset containing desired label for each patient. Choose from:"
            "Order, eid, Gender, Age.At.MHQ, PHQ9.Severity, Depressed.At.Baseline"
            "Neuroticism, Self.Harm.Ever, Not.Worth.Living, PCL.Score, GAD7.Severity"
        },
    )
    #Shira- needed for change in collect_fn. changed List to list 
    metadata_column_name_list: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of metadata columns to retain alongside the variable of interest."
        },
    )
    num_timepoints_per_voxel: int = field(
        default=490,
        metadata={
            "help": "Number of timepoints for each voxel given in 1 sample input to model. "
            "Must be divisible by timepoint_patching_size."
        },
    )
    timepoint_patching_size: int = field(
        default=49,
        metadata={
            "help": "Length of moving window of timepoints from each brain "
            "regions signal for 1 sample."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        self.data_files = None


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/image processor we are going to pre-train.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization.Don't set if you want to train a model from scratch."
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name_or_path"
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from s3"
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    hidden_size: int = field(default=128, metadata={"help": "Encoder hidden size."})
    num_hidden_layers: int = field(default=2, metadata={"help": "Encoder num layers."})
    num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in encoder."}
    )
    intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in encoder layers."}
    )
    decoder_hidden_size: int = field(
        default=128, metadata={"help": "Decoder hidden size."}
    )
    decoder_num_hidden_layers: int = field(
        default=2, metadata={"help": "Decoder num layers."}
    )
    decoder_num_attention_heads: int = field(
        default=2, metadata={"help": "Number of attention heads in the decoder."}
    )
    decoder_intermediate_size: int = field(
        default=512, metadata={"help": "Intermediate size in MLP in decoder layers."}
    )
    hidden_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "Dropout probability for layer activations in CellLM."},
    )
    attention_probs_dropout_prob: float = field(
        default=0.1,
        metadata={"help": "Dropout probability for attention coefficients in CellLM."},
    )
    mask_ratio: float = field(
        default=0.2,
        metadata={"help": "The ratio of the number of masked tokens per voxel."},
    )


@dataclass
class CustomTrainingArguments(TrainingArguments):
    remove_unused_columns: bool = field(
        default=False, metadata={"help": "Don't remove unused columns."}
    )
    do_train: int = field(default=True, metadata={"help": "Whether to do training."})
    do_eval: int = field(default=True, metadata={"help": "Whether to do eval."})
    #Shira- from gony
    skip_plots: bool = field(
        default=False, metadata={"help": "Enable debug mode (store true)."}
    )
    base_learning_rate: float = field(
        default=1e-3,
        metadata={
            "help": "Base learning rate: absolute_lr = base_lr * total_batch_size / 256."
        },
    )
    lr_scheduler_type: str = field(
        default="cosine_with_restarts",
        metadata={"help": "What learning rate scheduler to use."},
    )
    weight_decay: float = field(
        default=0.05,
        metadata={
            "help": "Weight decay (L2 regularization coefficient) for optimizer."
        },
    )
    num_train_epochs: int = field(
        default=100, metadata={"help": "Number of epochs to train for."}
    )
    warmup_ratio: float = field(
        default=0.05, metadata={"help": "Warmup ratio for learning rate scheduler."}
    )
    per_device_train_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during training."},
    )
    per_device_eval_batch_size: int = field(
        default=128,
        metadata={"help": "Batch size for each device used during evaluation."},
    )
    logging_strategy: str = field(
        default="steps",
        metadata={
            "help": "How often to log training metrics. If choose 'steps', specify logging_steps."
        },
    )
    logging_steps: int = field(
        default=10,
        metadata={
            "help": "If logging_strategy is 'steps', log training metrics every X iterations."
        },
    )
    evaluation_strategy: str = field(
        default="steps", metadata={"help": "How often to log eval results."}
    )
    eval_steps: int = field(
        default=10,
        metadata={
            "help": "If evaluation_strategy is 'steps', calculate validation metrics every X iterations."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "How often to save results and models."}
    )
    save_steps: int = field(
        default=10,
        metadata={
            "help": "If save_strategy is 'steps', save model checkpoint every X iterations."
        },
    )
    load_best_model_at_end: bool = field(
        default=True, metadata={"help": "At the end, load the best model."}
    )
    save_total_limit: int = field(
        default=50, metadata={"help": "Maximum number of models to save."}
    )
    seed: int = field(default=1234, metadata={"help": "Random seed."})
    wandb_logging: bool = field(
        default=False,
        metadata={
            "help": "Whether to log metrics to weights & biases during training."
        },
    )
    include_inputs_for_metrics: bool = field(
        default=True,
        metadata={
            "help": "Trainer will include model inputs in call to metrics calculation function. Depends on 'input_ids' being one of the input parameters to model, comes from tokenizer used? Currently incompatible with single-cell dataloader, leave as False."
        },
    )
    loss_fn: str = field(
        default="mse",
        metadata={"help": "Loss function for CellLM to use for pretraining."},
    )
    use_tanh_decoder: bool = field(
        default=False,
        metadata={
            "help": "If we want to use TanH as the nonlinearity for the output layer."
        },
    )




def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, CustomTrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Check that arguments for  make sense
    # Shira- took fix from gony
    if data_args.num_timepoints_per_voxel != -1:
        assert (
            data_args.num_timepoints_per_voxel % data_args.timepoint_patching_size == 0
        ), "Number of timepoints per voxel should be divisible by the timepoint patching size."

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    send_example_telemetry("run_mae", model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # --- Initialize Weights & Biases Logging ---#
    date_time_stamp = training_args.output_dir.split("/")[-1]

    if training_args.wandb_logging:
        # NOTE: Change project name to your own project name in your weights & biases account
        dataset_name = data_args.train_dataset_path.split("/")[-1]
        dataset_name = dataset_name.split("_cell_dataset")[0]
        wandb.init(
            project="BrainLM", name="{}-{}".format(dataset_name, date_time_stamp)
        )

    # --- Initialize Dataset ---#
    # Load arrow datasets
    train_ds = load_from_disk(data_args.train_dataset_path)
    print("train_ds.type:", type(train_ds))
    val_ds = load_from_disk(data_args.val_dataset_path)

    # Turn into a dictionary of Datasets
    ds = DatasetDict({"train": train_ds, "validation": val_ds})

    # Load gene information dataset (containing gene names, expression mean and std dev)


    # Shira
    # coords_ds = load_from_disk(data_args.coords_dataset_path)
    if '.dat' in str(data_args.coords_dataset_path):
        coords_ds = pd.read_csv('/home/ai_center/ai_users/gonyrosenman/students/users/troyansky1/BrainLM/toolkit/atlases/A424_Coordinates.dat',delimiter='\t', names=['idx', 'X', 'Y', 'Z'])
        coords_ds = Dataset.from_pandas(coords_ds)
    else:
        coords_ds = load_from_disk(data_args.coords_dataset_path)

    # Load model
    config_kwargs = {
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "use_auth_token": True if model_args.use_auth_token else None,
    }
    if model_args.config_name:
        config = BrainLMConfig.from_pretrained(model_args.config_name, **config_kwargs)
    elif model_args.model_name_or_path:
        config = BrainLMConfig.from_pretrained(
            model_args.model_name_or_path, **config_kwargs
        )
    else:
        config = BrainLMConfig(
            hidden_size=model_args.hidden_size,
            num_hidden_layers=model_args.num_hidden_layers,
            num_attention_heads=model_args.num_attention_heads,
            intermediate_size=model_args.intermediate_size,
            hidden_dropout_prob=model_args.hidden_dropout_prob,
            attention_probs_dropout_prob=model_args.attention_probs_dropout_prob,
            decoder_num_attention_heads=model_args.decoder_num_attention_heads,
            decoder_hidden_size=model_args.decoder_hidden_size,
            decoder_num_hidden_layers=model_args.decoder_num_hidden_layers,
            decoder_intermediate_size=model_args.decoder_intermediate_size,
            num_timepoints_per_voxel=data_args.num_timepoints_per_voxel,
            mask_ratio=model_args.mask_ratio,
            timepoint_patching_size=data_args.timepoint_patching_size,
            loss_fn=training_args.loss_fn,
            use_tanh_decoder=training_args.use_tanh_decoder,
        )
        logger.warning("You are instantiating a new config instance from scratch.")
        if model_args.config_overrides is not None:
            logger.info(f"Overriding config: {model_args.config_overrides}")
            config.update_from_string(model_args.config_overrides)
            logger.info(f"New config: {config}")

    # adapt config
    config.update({
        "mask_ratio": model_args.mask_ratio,
        "attention_probs_dropout_prob": model_args.attention_probs_dropout_prob
    })

    # create model
    if model_args.model_name_or_path:
        model = BrainLMForPretraining.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        logger.info("Training new model from scratch")
        model = BrainLMForPretraining(config)

    if training_args.wandb_logging:
        wandb.watch(model, log="all", log_freq=1)

    if data_args.recording_col_name is not None:
        recording_col_name = data_args.recording_col_name
    else:
        raise AttributeError(
            "Please specify the dataset column containing the signal recording (recording_col_name) in the DataTrainingArguments class."
        )

    if data_args.variable_of_interest_col_name is not None:
        variable_of_interest_col_name = data_args.variable_of_interest_col_name
    else:
        raise AttributeError(
            "Please specify the dataset column containing the signal recording (recording_col_name) in the DataTrainingArguments class."
        )
    # Shira- needed for change in collect_fn
    if data_args.metadata_column_name_list is not None:
        metadata_column_name_list = data_args.metadata_column_name_list
    else:
        raise AttributeError(
            "Please specify the metadata columns to retain (metadata_colufmn_name_list) in the DataTrainingArguments class."
        )

    # Shira
    if data_args.num_timepoints_per_voxel != -1:
        assert (
            data_args.num_timepoints_per_voxel % data_args.timepoint_patching_size == 0
        ), "Number of timepoints per voxel should be divisible by the timepoint patching size."

    # Shira- define encoders
    if metadata_column_name_list is not None and len(metadata_column_name_list) > 0:
        from datasets import concatenate_datasets
        from collections import defaultdict
        import numpy as np
        encoders = defaultdict(dict)
        decoders = defaultdict(dict)
        concatenated_ds = concatenate_datasets([dataset_ for dataset_ in ds.values()])
        for col in metadata_column_name_list:
            unique_values = np.unique(concatenated_ds[col])
            encoders[col] = {val: idx for idx, val in enumerate(unique_values)}
            decoders[col] = {idx: val for val, idx in encoders[col].items()}
    else:
        encoders, decoders = None, None
    training_args.skip_plots = training_args.skip_plots if is_main_process(training_args.local_rank) else True
    metrics_calculator = MetricsCalculator(training_args.skip_plots, encoders, decoders)



    def preprocess_fmri(examples):
        """
        Preprocessing function for dataset samples. This function is passed into Trainer as
        a preprocessor which takes in one row of the loaded dataset and constructs a model
        input sample according to the arguments which model.forward() expects.

        The reason this function is defined inside on main() function is because we need
        access to arguments such as cell_expression_vector_col_name.
        """
        # label = examples[variable_of_interest_col_name][0]
        # if math.isnan(label):
        #     label = -1  # replace nans with -1
        # else:
        #     label = int(label)
        
        # Shira's change from gony's coed!!
        if variable_of_interest_col_name in examples:
            labels = torch.tensor(examples[variable_of_interest_col_name], dtype=torch.float32)
            processed_labels = torch.nan_to_num(labels, nan=-1)
        else:
            # Fallback: use dummy labels
            processed_labels = torch.full((len(examples[recording_col_name]),), -1.0, dtype=torch.float32)


        #Shira (from gony)
        # Get the signal vectors and their scan durations
        signal_vectors = examples[recording_col_name]  # List of [scan_duration_i, num_regions]
        processed_signal_vectors = []
        for rec in signal_vectors:
            rec_tensor = torch.tensor(rec, dtype=torch.float32)  # shape: (T, num_voxels)
            scan_length = rec_tensor.shape[0]

            if data_args.num_timepoints_per_voxel == -1 or data_args.num_timepoints_per_voxel > scan_length:
                # Use the full scan length that still divides by patching_size
                largest_scan_length = data_args.timepoint_patching_size * (scan_length // data_args.timepoint_patching_size)
                start_idx = randint(0, scan_length - largest_scan_length)
                end_idx = start_idx + largest_scan_length
                signal_window = rec_tensor[start_idx:end_idx]
            else:
                # Randomly crop a window of length num_timepoints_per_voxel
                start_idx = randint(0, scan_length - data_args.num_timepoints_per_voxel)
                end_idx = start_idx + data_args.num_timepoints_per_voxel
                signal_window = rec_tensor[start_idx:end_idx, :]
                    # Transpose so that the output is (num_voxels, sequence_length) (i.e. 424 x sequence_length)
            signal_window = signal_window.transpose(0, 1)
            processed_signal_vectors.append(signal_window)


        # Add in key-value pairs for model inputs which CellLM is expecting in forward() function:
        #  signal_vectors and xyz_vectors
        #  These lists will be stacked into torch Tensors by collate() function (defined above).
        examples["signal_vectors"] = processed_signal_vectors

        # (Keep processing of xyz_vectors and labels unchanged)
        window_xyz_tensor = torch.tensor(
            [[c["X"], c["Y"], c["Z"]] for c in coords_ds], dtype=torch.float32
        )
        examples["xyz_vectors"] = [window_xyz_tensor] * len(processed_labels)
        examples["label"] = processed_labels
        return examples

    def collate_fn(examples):
        """
        This function tells the dataloader how to stack a batch of examples from the dataset.
        Need to stack gene expression vectors and maintain same argument names for model inputs
        which CellLM is expecting in forward() function:
            expression_vectors, sampled_gene_indices, and cell_indices
        """
        # Shira-used code from gony to try and avoid the error 
        #   IndexError: index 1 is out of bounds for dimension 0 with size 1

        signals = [example["signal_vectors"].permute(1,0) for example in examples]
        # print("signals:", (signals[0].shape), (signals[1].shape))

        padded_signals = torch.nn.utils.rnn.pad_sequence(signals, batch_first=True, padding_value=0)
        padded_signals = padded_signals.permute(0,2,1)
        # padding all the signals in all the batches to their maximum length (200)
        current_length = padded_signals.size(2)
        if current_length < 200:
            pad_width = 200 - current_length
            padded_signals = torch.nn.functional.pad(padded_signals, (0, pad_width), mode='constant', value=0)
        # print("padded_signals:", (padded_signals[0].shape), (padded_signals[1].shape))
        final_length = (padded_signals.shape[2] // data_args.timepoint_patching_size) * data_args.timepoint_patching_size
        signal_vectors = padded_signals[:, :, :final_length]

        metadata = {}
        for col in metadata_column_name_list:
            metadata[col] = torch.tensor([encoders[col][example[col]] for example in examples]).unsqueeze(-1)
        xyz_vectors = torch.stack([example["xyz_vectors"] for example in examples])
        # Shira change from GPT
        # labels = torch.stack([example["label"] for example in examples]).unsqueeze(-1)
        labels = torch.stack([example["label"] for example in examples]).unsqueeze(-1)


        # These inputs will go to model.forward(), names must match
        batch_dict = {
            "signal_vectors": signal_vectors,
            "xyz_vectors": xyz_vectors,
            "input_ids": signal_vectors,
            "labels": labels,
        }
        batch_dict.update(metadata)
        return batch_dict
        ###########################################
        # NOTE: What is input_ids, why is it here #
        ###########################################
        # We need signal_vectors vectors in metrics calculation.
        # The HuggingFace Trainer class only passes model outputs to the metrics callable,
        #  and the model output expected is hardcoded attributes, so we cannot modify the model output
        #  object to include signal_vectors.
        # Instead, if we pass input_ids as an argument into the model and set the Training argument
        #  include_inputs_for_metrics to True, then whatever tensor is in input_ids will be
        #  passed into metrics. Therefore, we pass signal_vectors as input_ids here.


    if training_args.do_train:
        if "train" not in ds:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            ds["train"] = (
                ds["train"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_train_samples))
            )
        # Set the training transforms
        ds["train"].set_transform(preprocess_fmri)

    if training_args.do_eval:
        if "validation" not in ds:
            raise ValueError("--do_eval requires a validation dataset")
        if data_args.max_eval_samples is not None:
            ds["validation"] = (
                ds["validation"]
                .shuffle(seed=training_args.seed)
                .select(range(data_args.max_eval_samples))
            )
        # Set the validation transforms
        ds["validation"].set_transform(preprocess_fmri)

    # Compute absolute learning rate
    total_train_batch_size = (
        training_args.train_batch_size
        * training_args.gradient_accumulation_steps
        * training_args.world_size
    )
    if training_args.base_learning_rate is not None:
        training_args.learning_rate = (
            training_args.base_learning_rate * total_train_batch_size / 256
        )
    # Shira- changed MetricsCalculator as well!!
    metrics_calculator = MetricsCalculator(training_args.skip_plots, encoders, decoders)

    # Initialize our trainer
    trainer = BrainLMTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"] if training_args.do_train else None,
        eval_dataset=ds["validation"] if training_args.do_eval else None,
        data_collator=collate_fn,
        compute_metrics=metrics_calculator,
    )

    # #Check if the dataset was created properly
    # print('len(ds["train"]): ',len(ds["train"])) #800
    # print('len(ds["validation"]):', len(ds["validation"])) #100
    # print(aaa)

    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            print("resume_from_checkpoint is not None-------------")
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            print("last_checkpoint is not None-------------")
            checkpoint = last_checkpoint
        print("else---------------------------")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()
        trainer.log_metrics("train", train_result.metrics)
        trainer.save_metrics("train", train_result.metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Write model card and (optionally) push to hub
    # kwargs = {
    #     "tasks": "masked-auto-encoding",
    #     "dataset": data_args.dataset_name,
    #     "tags": ["masked-auto-encoding"],
    # }
    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
