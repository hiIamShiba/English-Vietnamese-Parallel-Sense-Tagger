# English-Vietnamese Parallel Sense Tagger

## Introduction
[cite_start]This project implements an English-Vietnamese parallel sense tagger to solve the semantic ambiguity problem by exploiting bilingual data as an additional semantic signal[cite: 33, 47]. [cite_start]The model is built on the XLM-RoBERTa Base architecture with a Shared Encoder mechanism and Separate Classifiers for each language[cite: 58, 111].

## Directory Structure
- [cite_start]`dataset.jsonl`: The raw input dataset containing approximately 50,000 aligned and annotated English-Vietnamese parallel sentence pairs[cite: 62, 63].
- [cite_start]`en_sense_list.json` & `vi_sense_list.json`: Mapping dictionaries used to convert between sense labels (text format) and numerical identifiers (IDs)[cite: 64, 98].
- `sense_list_pipeline.ipynb`: The data processing pipeline for building the semantic dictionary lists.
- `rule_based_sense_tagger.ipynb`: A notebook containing the rule-based approach for sense tagging.
- `Finetune_XLM_R_Final_Introduce2NLP.ipynb`: The main source code for data preprocessing, formatting, and fine-tuning the XLM-RoBERTa model.
- `test_and_evaluate.py`: A Python script used to run tests and evaluate model performance on the Test set.
- [cite_start]`Report.pdf`: A detailed experimental report on the training setup process and analysis results[cite: 4, 60].

## Model Architecture
[cite_start]The system uses the `xlm-roberta-base` model as the base encoding component[cite: 113]:
- [cite_start]**Shared Encoder**: Extracts contextual feature vectors for both English and Vietnamese token sequences[cite: 114].
- [cite_start]**Separate Classifiers**: Two separate linear classification layers that map the feature vector space to the English and Vietnamese WordNet label spaces[cite: 116, 117, 118].
- [cite_start]**Loss Function**: Utilizes a composite loss function combining Cross-Entropy (for token classification) and Parallel Consistency Loss (calculated via Cosine similarity) with a coefficient of 0.1 to leverage bilingual semantic similarity[cite: 135, 137, 147].

## Data and Training
- [cite_start]**Data**: The dataset is randomly split into 80% for training and 20% for testing[cite: 65, 66]. [cite_start]The tokenization process applies the First-Token Tagging strategy for sub-words[cite: 126, 127].
- [cite_start]**Optimization**: Uses the AdamW algorithm combined with differential learning rate techniques (2x10^-5 for the Encoder and 1x10^-4 for the Classifiers)[cite: 171, 172, 173, 174].
- [cite_start]**Environment**: The system is deployed on the Google Colab platform using GPUs, the PyTorch library, and Transformers (HuggingFace)[cite: 82, 84, 88].

## Results
[cite_start]After training for 6 epochs, the model version achieving the most optimal metrics was recorded at the 4th epoch[cite: 160, 205]:
- [cite_start]**English F1-score**: 0.9690 [cite: 204]
- [cite_start]**Vietnamese F1-score**: 0.8981 [cite: 204]
- [cite_start]**Average F1-score**: 0.9336 [cite: 204]
