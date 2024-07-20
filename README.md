# Named Entity Recognition (NER) Project

## Project Description

This project focuses on Named Entity Recognition (NER) using a BERT-based model. NER is a key task in Natural Language Processing (NLP) that involves identifying and classifying entities in text into predefined categories such as names of people, organizations, locations, and other entities. The goal of this project is to develop a robust NER model using BERT and evaluate its performance on the CoNLL-2003 dataset.

## Methodology

### Model and Libraries

The model used in this project is based on the BERT (Bidirectional Encoder Representations from Transformers) architecture, which is well-suited for token classification tasks such as NER. The following libraries were utilized:

- `pandas`
- `numpy`
- `datasets`
- `transformers`
- `tensorflow`
- `spacy`
- `matplotlib`

### Data Preprocessing

The CoNLL-2003 dataset was used for training and evaluation. The dataset is publicly available and can be loaded using the `datasets` library from Hugging Face. The preprocessing steps involved tokenizing the text data and aligning NER tags with the tokens.

### Model Training

The BERT model was fine-tuned on the CoNLL-2003 dataset. The training configuration included adjusting hyperparameters such as learning rate, batch size, and the number of epochs. Despite resource limitations due to the free plan on Google Colab, the model achieved reasonable performance.

### Evaluation

The model was evaluated using precision, recall, F1 score, and accuracy metrics. The evaluation showed that the model was competitive with existing benchmarks, despite the constraints imposed by limited computational resources.

## Setup

To set up the environment, ensure you have Python installed. Then, install the required libraries using the `requirements.txt` file provided.

### Requirements

Create a `requirements.txt` file with the following content:

```plaintext
pandas
numpy
datasets
transformers
tensorflow
spacy
matplotlib

## Installation
Clone the repository and install the required libraries:
`
git clone https://github.com/Mody-Medhat/Named-Entity-Recognition-NER.git
cd NER-Project
pip install -r requirements.txt
`
