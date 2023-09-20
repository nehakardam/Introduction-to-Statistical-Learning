# Project Title: Language Identification in Twitter Data Using Machine Learning

## Project Description:
- Developed a language identification system for classifying Twitter text into one of nine Western European languages: English, Spanish, Portuguese, Galician, Basque, Catalan, French, Italian, and German.
- Implemented three different models for language identification: a Language-dependent Markov (n-gram) Model, a Recurrent Neural Network (RNN), and a Convolutional Neural Network (CNN).
- Preprocessed the Twitter text data, including creating character-level vocabularies, adding start and end tokens, and partitioning data by language.

## Modeling Options:
1. Language-dependent Markov (n-gram) Models:
   - Trained separate Markov models for each language using n-grams.
   - Explored various n-gram orders and smoothing methods.
   - Utilized the vocabulary established during data preprocessing.

2. Recurrent Neural Network (RNN):
   - Built a single RNN model that jointly models all languages using character and language embeddings.
   - Implemented the RNN architecture with options for GRU or LSTM cells.
   - Tuned hyperparameters, including input embedding sizes, recurrent layer size, and dropout rate, based on validation data.

3. Convolutional Neural Network (CNN):
   - Developed a CNN-based discriminative approach for language identification.
   - Employed convolutional layers with max pooling to process character sequences.
   - Tuned hyperparameters, such as the number of convolutional filters and output channels, based on validation accuracy.

## Validation and Evaluation:
- Utilized validation data to select the best-performing model and fine-tune hyperparameters.
- Evaluated model performance on the test data.
- Reported accuracy and macro F1 score as evaluation metrics for language identification.

## Implementation Details for RNN and CNN Models:
- Included character and language embeddings.
- Utilized recurrent layers (GRU or LSTM) for RNN.
- Employed convolutional layers with max pooling for CNN.
- Trained models to maximize log probability and minimize cross-entropy.
- Managed padding tokens and handled masking during loss computation.

## Key Achievements:
- Successfully implemented three different models for language identification.
- Conducted data preprocessing, hyperparameter tuning, and model training.
- Evaluated model performance using accuracy and macro F1 score.
- Demonstrated proficiency in machine learning and deep learning techniques for text classification.

## Skills Demonstrated:
- Machine Learning
- Deep Learning (RNN and CNN)
- Data Preprocessing
- Hyperparameter Tuning
- Model Evaluation
- Text Classification
- Sequence Modeling
