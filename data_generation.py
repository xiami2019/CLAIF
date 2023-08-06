import argparse
import openai
import os
from datetime import datetime

from modeling import CLAIFGenerator
from utils import set_seed, read_inputs, DatasetEntry

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory to which the generated dataset is saved")
    parser.add_argument("--generation_stage", type=str, default='stage-1', 
                        help='stage-1: generated sentence pairs; stage-2: generated similarity scores;')
    parser.add_argument("--llm_engine", type=str, default="text-davinci-003",
                        help="OpenAI's large language models.")
    parser.add_argument("--openai_api_key", type=str, required=True)
    parser.add_argument("--max_output_length", type=int, default=256,
                        help="The maximum output length for each generated text.")
    parser.add_argument("--top_p", type=float, default=0.9,
                        help="p value for top-p sampling (set to 0 to perform no top-p sampling)")
    parser.add_argument("--input_file", type=str,
                        help="An optional input file containing raw texts. This is required for generating text pair datasets.")
    parser.add_argument("--input_file_type", choices=["plain", "jsonl", "stsb"], default="jsonl",
                        help="The type of the input file. Choices are 'plain' (a raw text file with one input per line), 'jsonl' (a jsonl "
                             "file as produced by DINO) and 'stsb' (a TSV file in the STS Benchmark format)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="The batch size for generation (only if --input_file is not set)")
    parser.add_argument("--remove_identical_pairs", action='store_true',
                        help="Whether text pairs with text_a == text_b should be removed from the dataset (only for text pair datasets)")
    parser.add_argument("--allow_newlines_in_outputs", action='store_true',
                        help="If set to true, model outputs that contain a newline character before the end-of-sequence token (a quotation "
                             "mark) are not removed from the dataset.")
    parser.add_argument("--min_num_words", type=int, default=-1,
                        help="The minimum number of (whitespace-separated) words for each dataset entry. Entries with fewer words are "
                             "removed.")
    parser.add_argument("--using_cot", action="store_true", help='Zero-shot CoT, first generate analyze the difference between two sentences than give a score.')
    parser.add_argument("--temperature", type=float, default=0.7, help="temperature for GPT3 generation.")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    set_seed(args.seed)
    args.date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    print(f"Parameters: {args}")

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    assert args.input_file
    inputs = read_inputs(args.input_file, args.input_file_type)

    assert args.openai_api_key
    openai.api_key = args.openai_api_key

    generator = CLAIFGenerator(
        model=args.llm_engine,
        openai_api_key=args.openai_api_key,
        max_output_length=args.max_output_length,
        top_p=args.top_p,
        remove_identical_pairs=args.remove_identical_pairs,
        min_num_words=args.min_num_words,
        allow_newlines_in_outputs=args.allow_newlines_in_outputs,
        using_cot = args.using_cot,
        temperature = args.temperature
    )

    print("Starting dataset generation with CLAIF {}".format(args.generation_stage))
    outputs = generator.generate_dataset(inputs, batch_size=args.batch_size, generation_stage=args.generation_stage)

    print(f"Dataset generation complete, dataset contains {len(outputs)} entries")
    dataset_path = os.path.join(args.output_dir, 'generated-dataset.jsonl')
    DatasetEntry.save_list(outputs, dataset_path)
    print(f"Done saving dataset to file '{dataset_path}'")
