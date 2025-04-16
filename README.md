# IMDB-Sentiment-Classification-using-BERT

## 🧠 Project Overview

This project focuses on building a sentiment classification model using the IMDB movie review dataset.

### ✅ Key Highlights

- 📊 Processed and cleaned **75,000 movie reviews** to ensure high-quality input for training.
- 🔍 Fine-tuned a **Small BERT model** with custom encoder layers and an optimized classification head.
- ⚙️ Enhanced performance using **AdamW optimizer**, **learning rate tuning**, and **weight decay** to improve generalization.
- 📈 Achieved **72% training accuracy** and **75% validation accuracy**, demonstrating effective model fine-tuning and deep learning optimization.

The project showcases the power of transfer learning in NLP using BERT and TensorFlow.



This project demonstrates fine-tuning a **BERT** model on the **Stanford IMDB movie review dataset** for binary sentiment classification (positive/negative). The model is built using TensorFlow, TensorFlow Hub, and TensorFlow Text.

## 🚀 Features

- ✅ Downloads and preprocesses the [IMDB dataset](https://ai.stanford.edu/~amaas/data/sentiment/)
- ✅ Fine-tunes a **Small BERT model** (`bert_en_uncased_L-4_H-512_A-8`) from TensorFlow Hub
- ✅ Splits data into training, validation, and test sets
- ✅ Uses `EarlyStopping` and `ModelCheckpoint` callbacks
- ✅ Saves the model in `.h5`, `.keras`, and TensorFlow `SavedModel` formats

## 📦 Libraries Used

- TensorFlow
- TensorFlow Hub
- TensorFlow Text
- Keras

## 📁 Dataset Structure

- Training: `pos`, `neg`, and `unsup`
- Test: `pos`, `neg`
- Total files: ~100,000

## 🧠 Model Architecture

- BERT encoder (trainable)
- Dense(64) + Dropout(0.3)
- Final sigmoid output for binary classification

## 🏁 How to Run

```bash
pip install tf-models-official tensorflow tensorflow_hub tensorflow_text
