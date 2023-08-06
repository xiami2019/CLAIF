import math
import argparse
import random
from collections import defaultdict
from typing import List

from utils import DatasetEntry, DatasetEntryWithExp

def remove_explanations():
    pass

def postprocess_dataset(
        dataset: List[DatasetEntry],
        remove_identical_pairs: bool = True,
        remove_duplicates: bool = True,
        add_sampled_pairs: bool = True,
        max_num_text_b_for_text_a_and_label: int = 2,
        label_smoothing: float = 0.2,
        seed: int = 42,
        explanation = False
) -> List[DatasetEntry]:
    postprocessed_dataset = []
    num_text_b_for_text_a_and_label = defaultdict(int)

    rng = random.Random(seed)
    rng.shuffle(dataset)

    if remove_duplicates:
        dataset = list(set(dataset))

    for example in dataset:
        if remove_identical_pairs and example.text_a == example.text_b:
            continue

        if '<mask>' in example.text_a or '<mask>' in example.text_b:
            continue

        example.label = float(example.label) * (1 - label_smoothing) + (label_smoothing / 3 * 1.5)

        if max_num_text_b_for_text_a_and_label > 0:
            if num_text_b_for_text_a_and_label[(example.text_a, example.label)] >= max_num_text_b_for_text_a_and_label:
                continue
        postprocessed_dataset.append(DatasetEntry(text_a=example.text_a, text_b=example.text_b, label=example.label))
        num_text_b_for_text_a_and_label[(example.text_a, example.label)] += 1

    if add_sampled_pairs:
        sampled_dataset = []

        for text_a in set(x.text_a for x in postprocessed_dataset):
            for _ in range(2):
                text_b = rng.choice(postprocessed_dataset).text_b
                if explanation:
                    sampled_dataset.append(DatasetEntryWithExp(text_a=text_a, text_b=text_b, label=0, explanation='They are completely different in terms of the meaning and the words used.'))
                else:
                    sampled_dataset.append(DatasetEntry(text_a=text_a, text_b=text_b, label=0))

        postprocessed_dataset += sampled_dataset
    return postprocessed_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_file", type=str, required=True,
                        help="The input file which contains the STS dataset")
    parser.add_argument("--output_file", type=str, required=True,
                        help="The output file to which the postprocessed STS dataset is saved")

    args = parser.parse_args()

    ds = DatasetEntryWithExp.read_list(args.input_file)
    ds_pp = postprocess_dataset(ds, explanation=False)
    
    DatasetEntry.save_list(ds_pp, args.output_file)
