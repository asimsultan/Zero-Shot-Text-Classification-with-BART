
# Zero-Shot Text Classification with BART

Welcome to the Zero-Shot Text Classification with BART project! This project focuses on performing zero-shot text classification using the BART model.

## Introduction

Zero-shot text classification involves classifying text into predefined categories without needing any training data for those categories. In this project, we leverage the power of BART to perform zero-shot classification using a dataset of text samples.

## Dataset

For this project, we will use a custom dataset of text samples. You can create your own dataset and place it in the `data/classification_data.csv` file.

## Project Overview

### Prerequisites

- Python 3.6 or higher
- PyTorch
- Hugging Face Transformers
- Datasets
- Scikit-learn

### Installation

To set up the project, follow these steps:

```bash
# Clone this repository and navigate to the project directory:
git clone https://github.com/your-username/bart_zero_shot_classification.git
cd bart_zero_shot_classification

# Install the required packages:
pip install -r requirements.txt

# Ensure your data includes text samples and their labels. Place these files in the data/ directory.
# The data should be in a CSV file with two columns: text and label.

# To fine-tune the BART model for zero-shot text classification, run the following command:
python scripts/train.py --data_path data/classification_data.csv

# To evaluate the performance of the fine-tuned model, run:
python scripts/evaluate.py --model_path models/ --data_path data/classification_data.csv
