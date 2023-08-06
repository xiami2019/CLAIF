import csv
import json
import random
from typing import List, Optional, Any

import numpy as np

def set_seed(seed: int) -> None:
    """Set RNG seeds for python's `random` module, numpy and torch"""
    random.seed(seed)
    np.random.seed(seed)


def read_inputs(input_file: str, input_file_type: str) -> List[str]:
    valid_types = ['plain', 'jsonl', 'stsb']
    assert input_file_type in valid_types, f"Invalid input file type: '{input_file_type}'. Valid types: {valid_types}"

    if input_file_type == "plain":
        return read_plaintext_inputs(input_file)
    elif input_file_type == "jsonl":
        return read_jsonl_inputs_ab(input_file)
    elif input_file_type == "stsb":
        return read_sts_inputs(input_file)

def read_plaintext_inputs(path: str) -> List[str]:
    """Read input texts from a plain text file where each line corresponds to one input"""
    with open(path, 'r', encoding='utf8') as fh:
        inputs = fh.read().splitlines()
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs

def read_jsonl_inputs_ab(path: str) -> List[str]:
    """Read input texts from a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'"""
    ds_entries = DatasetEntry.read_list(path)
    print(f"Done loading {len(ds_entries)} inputs from file '{path}'")
    return [(entry.text_a, entry.text_b) for entry in ds_entries]


def read_jsonl_inputs(path: str) -> List[str]:
    """Read input texts from a jsonl file, where each line is one json object and input texts are stored in the field 'text_a'"""
    ds_entries = DatasetEntry.read_list(path)
    print(f"Done loading {len(ds_entries)} inputs from file '{path}'")
    return [entry.text_a for entry in ds_entries]


def read_sts_inputs(path: str) -> List[str]:
    """Read input texts from a tsv file, formatted like the official STS benchmark"""
    inputs = []
    with open(path, 'r', encoding='utf8') as fh:
        reader = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in reader:
            try:
                sent_a, sent_b = row[5], row[6]
                inputs.append(sent_a)
                inputs.append(sent_b)
            except IndexError:
                print(f"Cannot parse line {row}")
    print(f"Done loading {len(inputs)} inputs from file '{path}'")
    return inputs

def read_nli_inputs(path: str) -> List[str]:
    pass


class DatasetEntry:
    """This class represents a dataset entry for text (pair) classification"""

    def __init__(self, text_a: str, text_b: Optional[str], label: Any):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label

    def __repr__(self):
        if self.text_b is not None:
            return f'DatasetEntry(text_a="{self.text_a}", text_b="{self.text_b}", label={self.label})'
        else:
            return f'DatasetEntry(text_a="{self.text_a}", label={self.label})'

    def __key(self):
        return self.text_a, self.text_b, self.label

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, DatasetEntry):
            return self.__key() == other.__key()
        return False

    @staticmethod
    def save_list(entries: List['DatasetEntry'], path: str):
        with open(path, 'w', encoding='utf8') as fh:
            for entry in entries:
                fh.write(f'{json.dumps(entry.__dict__)}\n')

    @staticmethod
    def read_list(path: str) -> List['DatasetEntry']:
        pairs = []
        with open(path, 'r', encoding='utf8') as fh:
            for line in fh:
                pairs.append(DatasetEntry(**json.loads(line)))
        return pairs

class DatasetEntryWithExp:
    """This class represents a dataset entry for text (pair) classification"""

    def __init__(self, text_a, text_b, label, explanation):
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.explanation = explanation

    def __repr__(self):
        return f'DatasetEntryWithExplanation(text_a="{self.text_a}", text_b="{self.text_b}", label={self.label}, explanation={self.explanation})'

    def __key(self):
        return self.text_a, self.text_b, self.label, self.explanation

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, DatasetEntryWithExp):
            return self.__key() == other.__key()
        return False

    @staticmethod
    def save_list(entries: List['DatasetEntryWithExp'], path: str):
        with open(path, 'w', encoding='utf8') as fh:
            for entry in entries:
                fh.write(f'{json.dumps(entry.__dict__)}\n')

    @staticmethod
    def read_list(path: str) -> List['DatasetEntryWithExp']:
        pairs = []
        with open(path, 'r', encoding='utf8') as fh:
            for line in fh:
                pairs.append(DatasetEntryWithExp(**json.loads(line)))
        return pairs