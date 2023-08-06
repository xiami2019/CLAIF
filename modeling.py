import re
import math
import random
from typing import List, Optional, Union

import openai
from tqdm import tqdm
from utils import DatasetEntry, DatasetEntryWithExp

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def completion_with_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


class CLAIFGenerator:
    def __init__(self, model: str = None, openai_api_key: Optional[str] = None, max_output_length: int = 100, top_p: float = 0.9, remove_identical_pairs: bool = False, min_num_words: int = -1, \
                 allow_newlines_in_outputs: bool = False, using_cot: bool = False, temperature: float = 0.7):
        self.model = model
        self.openai_api_key = openai_api_key
        self.max_output_length = max_output_length
        self.top_p = top_p
        self.remove_identical_pairs = remove_identical_pairs
        self.min_num_words = min_num_words
        self.allow_newlines_in_outputs = allow_newlines_in_outputs
        self.using_cot = using_cot
        self.temperature = temperature

    def corrupt_sentence_with_mask(self, sentence, mask_rate, merge_adjacent_tokens=True, merge_rate=0.5):        
        mask_token = '<mask>'
        sentence = sentence.split(' ') if not isinstance(sentence, list) else None
        replace_token_number = math.ceil(len(sentence) * mask_rate)
        replace_token_index = random.sample(range(len(sentence)), replace_token_number)
        
        for i in replace_token_index:
            sentence[i] = mask_token

        # merge adjacent tokens
        if merge_adjacent_tokens and random.uniform(0,1) <= merge_rate:
            result = []
            for token in sentence:
                if token != '<mask>':
                    result.append(token)
                elif result == [] or result[-1] != '<mask>':
                    result.append(token)
            sentence = result

        sentence = ' '.join(sentence)
        
        return sentence

    def generate_dataset_with_explanation(self, input_texts: Optional[List[str]], batch_size: Optional[int] = None) -> List[DatasetEntryWithExp]:
        dataset = []

        for start_idx in tqdm(range(0, len(input_texts), batch_size)):
            inputs = input_texts[start_idx:start_idx+batch_size]
            current_generate_entries = self._generate_dataset_entries(inputs)
            if current_generate_entries == []:
                print('Insufficient balance')
                break
            dataset += current_generate_entries

        dataset = self._postprocess_dataset(dataset)
        return dataset

    def generate_dataset(self, input_texts: Optional[List[str]], batch_size: Optional[int] = None, generation_stage: str = 'stage-1') -> List[DatasetEntry]:

        def stage_1_generation():
            generate_with_inputs = input_texts is not None
            dataset = []
            mask_rates = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

            for start_idx in tqdm(range(0, len(input_texts), batch_size), desc="Generation step"):
                inputs = input_texts[start_idx:start_idx+batch_size]
                unmasked_inputs = []
                masked_inputs = []
                for sent in inputs:
                    for mask_rate in mask_rates:
                        unmasked_inputs.append(sent)
                        masked_inputs.append(self.corrupt_sentence_with_mask(sent, mask_rate=mask_rate))
                if 0.0 in mask_rates:
                    current_generate_entries = self._generate_dataset_entries_stage_1(unmasked_inputs, masked_inputs, first_no_mask=True)
                else:
                    current_generate_entries = self._generate_dataset_entries_stage_1(unmasked_inputs, masked_inputs)

                if current_generate_entries == []:
                    print('Insufficient balance')
                    break
                
                dataset += current_generate_entries

            dataset = self._postprocess_dataset_stage_1(dataset, generate_with_inputs)
            return dataset

        def stage_2_generation():
            dataset = []
            for start_idx in tqdm(range(0, len(input_texts), batch_size)):
                inputs = input_texts[start_idx:start_idx+batch_size]
                current_generate_entries = self._generate_dataset_entries_stage_2(inputs)
                if current_generate_entries == []:
                    print('Insufficient balance')
                    break
                dataset += current_generate_entries

            dataset = self._postprocess_dataset_stage_2(dataset)
            return dataset

        if generation_stage == 'stage-1':
            return stage_1_generation()
        elif generation_stage == 'stage-2':
            return stage_2_generation()

    def _generate_dataset_entries_stage_1(self, inputs: Union[str, int], masked_inputs: Union[str, int], first_no_mask=False) -> List[DatasetEntry]:
        
        if first_no_mask:
            instructions = [self._build_instruction_for_no_mask(masked_inputs[0])]
            for i in range(1, len(masked_inputs)):
                instructions.append(self._build_instruction(masked_inputs[i]))
        else:
            instructions = [self._build_instruction(masked_inputs[i]) for i in range(len(masked_inputs))]

        if self.openai_api_key is not None:
            try:
                model_responses = completion_with_backoff(
                    engine=self.model, prompt=instructions, max_tokens=self.max_output_length, top_p=self.top_p, temperature=self.temperature, stop=['"']
                )
                
                model_outputs = [model_response["text"] for model_response in model_responses['choices']]

            except openai.error.RateLimitError as e:
                print(e)
                return []

            except Exception as e: # something else went wrong
                print(e)
                return []
        else:
            raise Exception("No GPT3 key!")

        model_outputs = [
            self._process_output_stage_1(input_text=inputs[i], output_text=model_outputs[i], label=None)
            for i in range(len(model_outputs))
        ]

        model_outputs = [output for output in model_outputs if output is not None]
        return model_outputs

    def _generate_dataset_entries_stage_2(self, inputs: Union[str, int]) -> List[tuple]:

        if self.using_cot:
            instructions = [self._build_instruction_for_explanation_cot(inputs[i]) for i in range(len(inputs))]
        else:
            instructions = [self._build_instruction_for_explanation(inputs[i]) for i in range(len(inputs))]

        if self.openai_api_key is not None:
            try:
                model_responses = completion_with_backoff(
                    engine=self.model, prompt=instructions, max_tokens=self.max_output_length, top_p=self.top_p, temperature=self.temperature
                )
                
                model_outputs = [model_response["text"] for model_response in model_responses['choices']]

            except openai.error.RateLimitError as e:
                print(e)
                return []

            except Exception as e: # something else went wrong
                print(e)
                return []

        else:
            raise Exception("No GPT3 key!")

        model_outputs = [
            self._process_output_stage_2(sentence_a=inputs[i][0], sentence_b=inputs[i][1], instruction=instructions[i], output_text=model_outputs[i])
            for i in range(len(model_outputs))
        ]

        return model_outputs

    def _build_instruction(self, text: str) -> str:
        return "Replace all <mask> tokens in \"{}\" to make a new sentence. The new sentence is: \"".format(text)

    def _build_instruction_for_no_mask(self, text: str) -> str:
        return "Write two sentences that mean the same thing.\nSentence 1: \"{}\"\nSentence 2: \"".format(text)

    def _build_instruction_for_explanation_cot(self, text: str) -> str:
        sentence_a, sentence_b = text
        prompt = 'The semantic similarity score of two sentences is between 0.0 and 1.0, 0.0 means that the semantics are completely different and 1.0 means that the semantics are completely consistent.\nNow given two sentences \'{}\' and \'{}\', please explain the semantic difference between them and then give a semantic similarity score based on the semantic difference:\nThe semantic difference between these two sentences is'.format(sentence_a, sentence_b)
        return prompt
    
    def _build_instruction_for_explanation(self, text: str) -> str:
        sentence_a, sentence_b = text
        prompt = 'The similarity score for two sentences is in the range from 0.0 to 1.0, 0.0 means completely different and 1.0 means almost the same.\nNow give two sentences \'{}\' and  \'{}\', please give a similarity score of these two sentences and give the reason:\nThe similarity score for these two sentences is'.format(sentence_a, sentence_b)
        return prompt
    
    def _process_output_stage_1(self, input_text: Union[str, int], output_text: str, label: str) \
            -> Optional[DatasetEntry]:
        return DatasetEntry(text_a=input_text, text_b=output_text , label=label)

    def _process_output_stage_2(self, sentence_a, sentence_b, instruction, output_text):
        return (sentence_a, sentence_b, instruction, output_text)

    def _postprocess_dataset_stage_1(self, dataset: List[DatasetEntry], generate_with_inputs: bool) -> List[DatasetEntry]:
        if self.min_num_words > 0:
            if generate_with_inputs:
                dataset = [entry for entry in dataset if len(entry.text_b.split()) >= self.min_num_words]
            else:
                dataset = [entry for entry in dataset if len(entry.text_a.split()) >= self.min_num_words]

        if generate_with_inputs and self.remove_identical_pairs:
            dataset = [entry for entry in dataset if entry.text_a != entry.text_b]

        return dataset

    def _postprocess_dataset_stage_2(self, dataset):
        '''
        split similarity score and explanation
        '''
        pattern = re.compile(r'[0-9\.]*[0-9]')
        new_dataset = []
        invalid_explanation = 0
        for sample in dataset:
            sentence_a, sentence_b, instruction, output_text = sample
            res = re.findall(pattern, output_text)
            if len(res) == 0:
                invalid_explanation += 1
                continue
            if self.using_cot:
                similarity_score = res[-1]
            else:
                similarity_score = res[0]
            if self.using_cot:
                new_dataset.append(DatasetEntryWithExp(sentence_a, sentence_b, similarity_score, "The semantic difference between these two sentences is" + output_text))
            else:
                new_dataset.append(DatasetEntryWithExp(sentence_a, sentence_b, similarity_score, "The similarity score for these two sentences is" + output_text))
        print("Invalid explanation number is: {}".format(invalid_explanation))

        return new_dataset