from typing import Optional
import math
import os
import csv
import json
import numpy as np
import re

from argdantic import ArgParser
from pydantic import BaseModel
from tqdm import tqdm
from huggingface_hub import hf_hub_download

from common import PuzzleDatasetMetadata, dihedral_transform


CHARSET = "# SGo"

UNK_ID, PAD_ID, SOS_ID, EOS_ID, SEP_ID = 0, 1, 2, 3, 4

IGNORE_LOSS_ID = -100

cli = ArgParser()

VOCABULARY = {"opposite": 1, "and": 2, "after": 3, "I_RUN": 4, "look": 5, "I_LOOK": 6, "run": 7, "turn": 8, "right": 9, "thrice": 10, "left": 11, "I_WALK": 12, "jump": 13, "I_TURN_RIGHT": 14, "walk": 15, "I_TURN_LEFT": 16, "I_JUMP": 17, "twice": 18, "around": 19, "[PAD]" : 20}

class DataProcessConfig(BaseModel):
    source_repo: str = "sapientinc/maze-30x30-hard-1k"
    output_dir: str = "data/SCAN"

    subsample_size: Optional[int] = None
    aug: bool = False


def convert_subset(set_name: str, config: DataProcessConfig, path):
    # Read CSV
    grid_size = None
    inputs = []
    labels = []
    
    pattern = r"IN: (.*)OUT: (.*)"
    with open(path, "r") as f:
        content = f.readlines()

        for line in content:
            match = re.search(pattern, line)
            inputs.append([VOCABULARY[e] for e in match.group(1).split()])
            labels.append([VOCABULARY[e] for e in match.group(2).split()])

    max_seq_len = max(max(len(i) for i in inputs), max(len(l) for l in labels))

    inputs = [[inp[i] if i < len(inp) else VOCABULARY["[PAD]"] for i in range(max_seq_len)] for inp in inputs]
    labels = [[lab[i] if i < len(lab) else -100 for i in range(max_seq_len)] for lab in labels]


    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    if set_name == "train" and config.subsample_size is not None:
        total_samples = len(inputs)
        if config.subsample_size < total_samples:
            indices = np.random.choice(total_samples, size=config.subsample_size, replace=False)
            inputs = [inputs[i] for i in indices]
            labels = [labels[i] for i in indices]

    # Generate dataset
    results = {k: [] for k in ["inputs", "labels", "puzzle_identifiers", "puzzle_indices", "group_indices"]}
    puzzle_id = 0
    example_id = 0
    
    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    print(max_seq_len)

    # inp: list of input_ids
    # out: list of labels (tgt_ids)
    
    for inp, out in zip(tqdm(inputs), labels):
        # Dihedral transformations for augmentation
        for aug_idx in range(8 if (set_name == "train" and config.aug) else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1
            
            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)
            
        # Push group
        results["group_indices"].append(puzzle_id)

    print(np.array(results["inputs"]).shape)

    # To Numpy
    def _seq_to_numpy(seq):

        arr = np.array(seq)
        return arr
        # return np.expand_dims(arr, axis=0)  
    
    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    # Metadata
    metadata = PuzzleDatasetMetadata(
        seq_len=max_seq_len,  # type: ignore
        vocab_size=len(VOCABULARY)+1,  # PAD + Charset
        pad_id=0,
        ignore_label_id=0,
        blank_identifier_id=0,
        num_puzzle_identifiers=1,
        total_groups=len(results["group_indices"]) - 1,
        mean_puzzle_examples=1,
        total_puzzles=len(results["group_indices"]) - 1,
        sets=["all"]
    )

    # Save metadata as JSON.
    save_dir = os.path.join(config.output_dir, set_name)
    os.makedirs(save_dir, exist_ok=True)

    # print(metadata.model_dump())
    print(results)

    with open(os.path.join(save_dir, "dataset.json"), "w") as f:
        json.dump(metadata.model_dump(), f)
        
    # Save data
    for k, v in results.items():
        np.save(os.path.join(save_dir, f"all__{k}.npy"), v)
        
    # Save IDs mapping (for visualization only)
    with open(os.path.join(config.output_dir, "identifiers.json"), "w") as f:
        json.dump(["<blank>"], f)


@cli.command(singleton=True)
def preprocess_data(config: DataProcessConfig):
    convert_subset("train", config, "/home/victor-moene/master/SCAN/simple_split/tasks_train_simple.txt")
    convert_subset("test", config, "/home/victor-moene/master/SCAN/simple_split/tasks_test_simple.txt")


if __name__ == "__main__":
    cli()
