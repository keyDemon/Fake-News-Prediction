# Fake News Prediction

This project builds a machine learning model to classify news articles as either **real** or **fake** using logistic regression. By pre-processing text data and vectorizing it, the model can detect patterns and characteristics in the text that indicate whether the content is authentic or fabricated.

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Dependencies](#dependencies)
- [Data Pre-processing](#data-pre-processing)
- [Model Training](#model-training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)

## Project Overview
This project utilizes natural language processing (NLP) and machine learning techniques to detect fake news. The primary steps include data cleaning, text processing (such as stemming and removing stop words), transforming text data into numerical form, and training a logistic regression model on the processed data.

## Dataset
The dataset used in this project, `train.csv`, contains labeled news articles with columns for author, title, and a label indicating whether the news is fake (1) or real (0).

## Dependencies
Make sure you have the following Python libraries installed:
- `numpy`
- `pandas`
- `nltk`
- `sklearn`

You can install these packages using:
```bash
pip install numpy pandas nltk scikit-learn
```

## Data Pre-processing
1. **Loading the Data**: Load the dataset and fill any missing values with empty strings.
2. **Text Processing**: Combine the `author` and `title` fields into a single `content` field.
3. **Stemming**: Apply stemming to reduce each word to its root form, helping to normalize the text.
4. **Stop Words Removal**: Remove commonly used words that do not add significant value to model predictions, like "the" and "is".

## Model Training
The model used here is **Logistic Regression**, a classification algorithm that works well for text classification tasks.

1. **Text Vectorization**: Use `TfidfVectorizer` to convert text data into numerical vectors.
2. **Splitting the Data**: The data is split into training and test sets (80/20 split) using `train_test_split` from `sklearn`.
3. **Training**: Train a logistic regression model on the training data.

## Evaluation
The model's accuracy is assessed using the accuracy score metric:
- **Training Accuracy**: Measures how well the model fits the training data.
- **Test Accuracy**: Measures the model's performance on unseen data.

## Usage
To use the model for predictions:

1. Run the code to load and preprocess the dataset.
2. Train the model.
3. Test the model with a sample input to see if it classifies the news as real or fake. 

Example:
```python
X_new = X_test[0]
prediction = model.predict(X_new)
if prediction:
    print("The news is fake")
else:
    print("The news is real")
```

## Results
The final model provides accuracy scores for both training and test datasets, allowing you to gauge its performance. 

## Acknowledgments
Thanks to the dataset provider and the NLP community for making tools available to develop such models. 
