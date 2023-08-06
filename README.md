# Improving Contrastive Learning of Sentence Embeddings from AI Feedback

This repository contains the code, data and model checkpoints for our paper [Improving Contrastive Learning of Sentence Embeddings from AI
Feedback](https://arxiv.org/abs/2305.01918) (CLAIF).

## Data Generation
You can choose your own data as original sentences to construct datasets for sentence embeddings learning. Here we use a small demo set of sentences for example.  
We use **text-davinci-003** as the default engine.
### Stage Overview
![](pics/generation_process.png)
### Step 1: Sentence Pair Generation
```python
python data_generation.py --generation_stage stage-1 --output_dir sentence_pairs --input_file demo_sentences.csv --input_file_type stsb --batch_size 2 --openai_api_key <your_openai_api_key>
```
After step1, you will get sentence pairs in 'sentence_pairs/generated-dataset.jsonl' with a jsonl format.

### Step 2: Semantic Similarity Labeling
We use **text-davinci-003** as default.
```python
python data_generation.py --generation_stage stage-2 --output_dir sentence_pairs_with_labels --input_file ./sentence_pairs/generated-dataset.jsonl --input_file_type jsonl --batch_size 5 --openai_api_key <your_openai_api_key>
```
After step2, you will get sentence pairs with similarity scores and explainations from AI feedback in 'sentence_pairs_with_labels/generated-dataset.jsonl' with a jsonl format.