# Secret Leak Detection in Software Issue Reports using LLMs: A Comprehensive Evaluation

## Overview

This repository contains the replication package for our paper **"Secret Leak Detection in Software Issue Reports using LLMs: A Comprehensive Evaluation"**. This work presents a comprehensive study on detecting accidentally exposed secrets (API keys, tokens, credentials) in GitHub issue reports using Large Language Models.

## Abstract

In the digital era, accidental exposure of sensitive information such as API keys, tokens, and credentials is a growing security threat. While most prior work focuses on detecting secrets in source code, leakage in software issue reports remains largely unexplored. This study fills that gap through a large-scale analysis and a practical detection pipeline for exposed secrets in GitHub issues. 

Our pipeline combines regular expression–based extraction with large language model (LLM)–based contextual classification to detect real secrets and reduce false positives. We build a benchmark of **54,148 instances** from public GitHub issues, including **5,881 manually verified true secrets**. Using this dataset, we evaluate entropy-based baselines and keyword heuristics used by prior secret detection tools, classical machine learning, deep learning, and LLM-based methods.

### Key Findings

- **Regex and entropy-based approaches** achieve high recall but poor precision
- **Smaller models** (CodeBERT, XL-Net) greatly improve performance (**F1 = 92.70%**)
- **Proprietary models** (GPT-4o) perform moderately in few-shot settings (**F1 = 80.13%**)
- **Fine-tuned open-source LLMs** (Qwen, LLaMA) achieve up to **94.49% F1**
- **Real-world validation** on 178 GitHub repositories achieves **F1 = 0.82**

## Repository Structure

```
├── Data/                          # Dataset files
│   ├── main_data.csv             # Main dataset with 54,148 instances
│   ├── train.csv                 # Training split
│   ├── val.csv                   # Validation split
│   ├── test.csv                  # Test split
│   └── test_wild.csv             # In-the-wild test data (178 repos)
│
├── Data-Handling/                 # Data collection and preprocessing
│   ├── Dataset_Generation_labelled.ipynb
│   ├── inspect_labelled_reports.ipynb
│   ├── Fetching/                 # Data collection scripts
│   │   ├── crawler-1.py
│   │   ├── crawler-2.py
│   │   ├── fetch_real_repo_issues_and_scan.py
│   │   └── github_repo_urls.txt
│   └── Analysis/                 # Data analysis and visualization
│       ├── analyze_trends.py
│       ├── custom_visualizations.py
│       └── visualize_trends.py
│
├── Baselines/                     # Baseline methods
│   ├── regex+entropy.ipynb       # Regex + Entropy baseline
│   ├── regex+entropy+keyword+heuristic.ipynb
│   ├── handcrafted_features_classifier.py
│   └── textcnn_ensemble.py       # TextCNN ensemble model
│
├── Model/                         # LLM-based detection models
│   ├── bert_training.ipynb       # BERT/CodeBERT training
│   ├── bert_inference.ipynb
│   ├── llama_training.ipynb      # LLaMA fine-tuning
│   ├── llama_inference.ipynb
│   ├── qwen_training.ipynb       # Qwen fine-tuning
│   ├── qwen_inference.ipynb
│   ├── mistral_training.ipynb    # Mistral fine-tuning
│   ├── mistral_inference.ipynb
│   ├── gemma_training.ipynb      # Gemma fine-tuning
│   ├── gemma_inference.ipynb
│   ├── deepseek_training.ipynb   # DeepSeek fine-tuning
│   ├── deepseek_inference.ipynb
│   ├── gpt.py                    # GPT-4o evaluation
│   └── gemini.py                 # Gemini evaluation
│
├── Results/                       # Experimental results
│   ├── regex_entropy_predictions.csv
│   ├── regex_entropy_keyword_heuristic_predictions.csv
│   ├── decision_tree.csv
│   ├── logistic_regression.csv
│   ├── naive_bayes.csv
│   ├── random_forest.csv
│   ├── svm.csv
│   ├── k-nearest_neighbors.csv
│   ├── llama.csv                 # LLaMA fine-tuned results
│   ├── qwen.csv                  # Qwen fine-tuned results
│   ├── mistral.csv               # Mistral fine-tuned results
│   ├── gpt4o_zs.csv              # GPT-4o zero-shot results
│   ├── gpt4o_fs.csv              # GPT-4o few-shot results
│   ├── gemini_zs.csv             # Gemini zero-shot results
│   ├── gemini_fs.csv             # Gemini few-shot results
│   ├── albert-base-v2test_predictions.csv
│   ├── bert-base-casedtest_predictions.csv
│   ├── bert-base-uncasedtest_predictions.csv
│   ├── distilbert-base-casedtest_predictions.csv
│   ├── distilbert-base-uncasedtest_predictions.csv
│   ├── funnel-transformer_mediumtest_predictions.csv
│   ├── google_bigbird-roberta-basetest_predictions.csv
│   ├── google_electra-base-discriminatortest_predictions.csv
│   ├── microsoft_codebert-basetest_predictions.csv
│   ├── roberta-basetest_predictions.csv
│   ├── xlnet-base-casedtest_predictions.csv
│   └── test_metrics.txt          # Aggregated metrics for all models
│
└── Survey/                        # Motivational survey data
    ├── Motivation Survey.pdf     # Survey questionnaire and responses
    └── Motivational survey.xlsx  # Survey data in spreadsheet format
```

## Requirements

### Software Dependencies

```bash
# Python 3.8+
pip install torch transformers pandas numpy scikit-learn
pip install jupyter notebook
pip install matplotlib seaborn

# For proprietary models
pip install openai google-generativeai
```

## Dataset

Our dataset contains 54,148 instances extracted from public GitHub issues:
- **5,881 manually verified true secrets**
- **48,267 false positives** (strings that look like secrets but aren't)
- Covers multiple secret types: API keys, tokens, passwords, credentials, etc.
- You can access the dataset [here](https://drive.google.com/drive/u/0/folders/1QQ9XltpERkJre-vYXWhSQUYDPg17cvXB). 

The dataset is split into:
- Training set: `Data/train.csv`
- Validation set: `Data/val.csv`
- Test set: `Data/test.csv`
- In-the-wild test set: `Data/test_wild.csv` (178 real repositories)

## Replication Instructions

### Baseline Methods

#### Regex + Entropy
```bash
jupyter notebook Baselines/regex+entropy.ipynb
```

#### Regex + Entropy + Keywords + Heuristics
```bash
jupyter notebook Baselines/regex+entropy+keyword+heuristic.ipynb
```

#### Handcrafted Features Classifier
```bash
python Baselines/handcrafted_features_classifier.py
```

#### TextCNN Ensemble
```bash
python Baselines/textcnn_ensemble.py
```

### Fine-tuned Models

#### BERT/CodeBERT
```bash
# Training
jupyter notebook Model/bert_training.ipynb

# Inference
jupyter notebook Model/bert_inference.ipynb
```

#### Open-Source LLMs (LLaMA, Qwen, Mistral, Gemma, DeepSeek)

Each model has a training and inference notebook:
```bash
# Example for LLaMA
jupyter notebook Model/llama_training.ipynb
jupyter notebook Model/llama_inference.ipynb

# Similar for Qwen, Mistral, Gemma, DeepSeek
```

### Proprietary Models (Few-Shot)

#### GPT-4o
```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"
python Model/gpt.py
```

#### Gemini
```bash
# Set your Google API key
export GOOGLE_API_KEY="your-api-key"
python Model/gemini.py
```

## Results

| Model Type | Model | F1 Score | Precision | Recall |
|-----------|-------|----------|-----------|--------|
| Baseline | Regex + Entropy | - | Low | High |
| Small Model | CodeBERT | 92.70% | - | - |
| Small Model | XL-Net | 92.70% | - | - |
| Proprietary (Few-shot) | GPT-4o | 80.13% | - | - |
| Fine-tuned LLM | Qwen | 94.49% | - | - |
| Fine-tuned LLM | LLaMA | 94.49% | - | - |
| **In-the-Wild** | Best Model | **82.00%** | - | - |

Detailed prediction results for all models are available in the `Results/` directory. Each CSV file contains the model's predictions on the test set with corresponding ground truth labels and confidence scores. - You can access the results [here](https://drive.google.com/drive/u/0/folders/1QQ9XltpERkJre-vYXWhSQUYDPg17cvXB). 

## Data Collection

### Fetching GitHub Issues

The repository includes scripts to collect GitHub issues:

```bash
# Crawler scripts
python Data-Handling/Fetching/crawler-1.py
python Data-Handling/Fetching/crawler-2.py

# Fetch and scan real repositories
python Data-Handling/Fetching/fetch_real_repo_issues_and_scan.py
```

### Data Analysis

```bash
# Analyze trends in the dataset
python Data-Handling/Analysis/analyze_trends.py

# Generate visualizations
python Data-Handling/Analysis/visualize_trends.py
python Data-Handling/Analysis/custom_visualizations.py
```

## Detection Pipeline

Our detection pipeline consists of two stages:

1. **Extraction Stage**: Use regex patterns to extract potential secrets from issue text
2. **Classification Stage**: Use LLMs to classify whether extracted strings are real secrets

This two-stage approach significantly reduces false positives while maintaining high recall.

## Citation

If you use this code or data in your research, please cite our paper:

```
@misc{wahab2024secretbreachpreventionsoftware,
      title={Secret Breach Prevention in Software Issue Reports}, 
      author={Zahin Wahab and Sadif Ahmed and Md Nafiu Rahman and Rifat Shahriyar and Gias Uddin},
      year={2024},
      eprint={2410.23657},
      archivePrefix={arXiv},
      primaryClass={cs.SE},
      url={https://arxiv.org/abs/2410.23657}, 
}
```

---

**Note**: This repository is for research purposes. The dataset contains only publicly available information from GitHub issues that were already exposed at the time of collection.

