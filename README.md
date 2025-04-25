# IMDb Genre Classification

## 📑 Project Overview

This project aims to **classify movies into multiple genres** based on their **plot descriptions** using machine learning techniques. It uses a **multi-label classification** approach where each movie can belong to more than one genre. The project utilizes **TF-IDF** vectorization and **Logistic Regression** (with One-vs-Rest) as the baseline model.

The dataset includes four text files:

- `train_data.txt` – Training data containing movie IDs, titles, genres, and descriptions.
- `test_data.txt` – Test data containing movie IDs, titles, and descriptions (without genres).
- `test_data_solution.txt` – The correct genres for the movies in `test_data.txt`.
- `description.txt` – Provides more information about the dataset.

---

## 🎯 Task Objectives

The key objectives of this project are:

- **Data Ingestion:** Parse custom-formatted `.txt` files with the `:::` delimiter.
- **Text Vectorization:** Convert raw movie descriptions into numerical representations using **TF-IDF**.
- **Model Training:** Use **Logistic Regression (One-vs-Rest)** for multi-label classification.
- **Evaluation:** Measure model performance using **F1 score**, **Hamming loss**, and **accuracy**.
- **Misclassification Analysis:** Identify and analyze the most common misclassifications.

---

## 🚀 Steps to Run the Project

### 1. **Upload Required Files**

Upload the following 4 files from Kaggle into Google Colab:

- `train_data.txt`
- `test_data.txt`
- `test_data_solution.txt`
- `description.txt` (optional)

You can upload them using the **drag-and-drop** feature in Google Colab or the code below:

```python
from google.colab import files
uploaded = files.upload()  # Upload all 4 files: train_data.txt, test_data.txt, test_data_solution.txt, description.txt
```

---

### 2. **Install Required Libraries**

Run the following command to install the required packages:

```python
!pip install -q scikit-multilearn
```

---

### 3. **Load and Preprocess the Data**

Load the dataset using the custom `:::` delimiter format. The following code will parse the files into pandas DataFrames:

```python
import pandas as pd

def load_custom_file(filepath, has_genre=True):
    data = []
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(":::")
            if has_genre and len(parts) == 4:
                data.append({'ID': parts[0], 'Title': parts[1], 'Genre': parts[2], 'Description': parts[3]})
            elif not has_genre and len(parts) == 3:
                data.append({'ID': parts[0], 'Title': parts[1], 'Description': parts[2]})
    return pd.DataFrame(data)

train_df = load_custom_file("train_data.txt", has_genre=True)
test_df = load_custom_file("test_data.txt", has_genre=False)
solution_df = load_custom_file("test_data_solution.txt", has_genre=True)

# Clean genres
train_df['Genre'] = train_df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
solution_df['Genre'] = solution_df['Genre'].apply(lambda x: [g.strip() for g in x.split(',')])
```

---

### 4. **Feature Extraction (TF-IDF)**

Transform movie plot descriptions into numerical features using **TF-IDF** (Term Frequency-Inverse Document Frequency) vectorization:

```python
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train = vectorizer.fit_transform(train_df['Description'])
X_test = vectorizer.transform(test_df['Description'])
```

---

### 5. **Label Binarization**

Since the problem is a **multi-label classification**, we need to convert the genre labels into a binary matrix:

```python
from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(train_df['Genre'])
y_test = mlb.transform(solution_df['Genre'])
```

---

### 6. **Train the Model**

We will train a **Logistic Regression** model using the **One-vs-Rest** strategy, where each genre gets its own classifier.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', max_iter=1000))
clf.fit(X_train, y_train)
```

---

### 7. **Evaluate the Model**

After training the model, evaluate its performance using the following metrics:

- **F1 score (micro):** Measures overall accuracy for multi-label classification.
- **Hamming loss:** Fraction of incorrectly predicted labels.
- **Subset accuracy:** Exact match ratio between predicted and actual genres.

```python
from sklearn.metrics import f1_score, hamming_loss, accuracy_score, classification_report

y_pred = clf.predict(X_test)
print("F1 Score (micro):", f1_score(y_test, y_pred, average='micro'))
print("Hamming Loss:", hamming_loss(y_test, y_pred))
print("Accuracy (subset):", accuracy_score(y_test, y_pred))
```

---

### 8. **Visualize Genre-wise Performance**

Visualize the F1 scores for each genre using a bar plot:

```python
import matplotlib.pyplot as plt
import pandas as pd

report = classification_report(y_test, y_pred, target_names=mlb.classes_, output_dict=True)
pd.DataFrame(report).T.iloc[:-3][['f1-score']].sort_values('f1-score').plot(
    kind='barh', figsize=(10, 8), title="Genre-wise F1 Scores"
)
```

---

### 9. **Analyze Misclassifications**

You can inspect some **misclassified movie titles** to understand where the model needs improvement:

```python
import numpy as np

wrong_idx = np.where((y_pred != y_test).any(axis=1))[0][:5]
for idx in wrong_idx:
    print(f"Title: {test_df.iloc[idx]['Title']}")
    print(f"Predicted: {mlb.inverse_transform(y_pred[idx].reshape(1, -1))[0]}")
    print(f"Actual:    {solution_df.iloc[idx]['Genre']}\n")
```

---

## 📝 Code Quality Guidelines

- **Modular:** The code is broken into reusable functions for clarity.
- **Clean:** The flow from data loading → processing → model training → evaluation is logically structured.
- **Commented:** Inline comments are provided at each critical step to help understand the code.
- **Extendable:** The code is designed to be easily extended, allowing for the introduction of advanced techniques like **BERT embeddings**, **deep learning**, or more sophisticated classifiers.

---

## 🔧 Next Steps

1. **Fine-tune the model:** Experiment with hyperparameters of the Logistic Regression model or try other models like **XGBoost**, **LightGBM**, or **SVM**.
2. **Use advanced feature extraction:** Implement **word embeddings** (Word2Vec, GloVe) or even **BERT** for better representation of text.
3. **Deploy the model:** Once optimized, deploy the model for real-time predictions on new movie descriptions.

---

### 🚀 Final Thoughts

This project provides a great starting point for multi-label text classification. The current model can be used as a baseline, and further improvements can be made with more advanced techniques such as **deep learning** or **transformers**. It's an excellent way to gain insights into both **text preprocessing** and **multi-label classification**.
