
# News Text Classification 

## 📌 Overview  
This project focuses on classifying news articles into four categories: **World**, **Sports**, **Business**, and **Sci/Tech**. It applies traditional machine learning models like **K-Nearest Neighbors** and **Naive Bayes**, as well as **deep learning CNNs** with and without pre-trained GloVe embeddings. The dataset contains news headlines (`Title`), article content (`Description`), and corresponding class labels.

---

## 📂 Files & Structure

```
├── train.csv                 # Training dataset
├── test.csv                  # Test dataset
├── glove.6B.50d.txt          # GloVe pre-trained word vectors (50D)
├── glove.6B.100d.txt         # GloVe pre-trained word vectors (100D)
└── assignment_script.py     # Main implementation
```

---

## 🔧 Features & Workflow

### 🧹 1. Data Preparation
- Combines `Title` and `Description` into a single `Text` field.
- Applies preprocessing:
  - Lowercasing
  - Removing punctuation and stopwords
  - Removing alphanumeric noise

### 🧠 2. Classical Machine Learning Models
- **Vectorization** using:
  - `CountVectorizer`
  - `TfidfVectorizer`
- **Models used:**
  - `KNeighborsClassifier`
  - `MultinomialNB` (Naive Bayes)

### 🤖 3. Deep Learning Models (CNN)
- Tokenization using Keras `Tokenizer`
- Padding to uniform sequence lengths
- Three CNN architectures:
  1. **Baseline CNN:** Random embeddings
  2. **GloVe CNN:** Uses GloVe word embeddings (50D or 100D)
  3. **Fine-Tuned CNN:** GloVe + Regularization, Dropout, tuned hyperparameters

### 📈 4. Evaluation
- Accuracy, Precision, Recall, F1-Score
- Confusion Matrix Heatmap (Matplotlib + Seaborn)
- Training/Validation curves per model

---

## 🧪 Results Summary (example)

| Model                    | Vectorizer | Test Accuracy |
|-------------------------|------------|---------------|
| KNN                     | Count      | ~0.79         |
| Naive Bayes             | TF-IDF     | ~0.82         |
| CNN (Random Embedding)  | Tokenizer  | ~0.88         |
| CNN (GloVe Embedding)   | GloVe 100D | ~0.91         |
| CNN (Fine-Tuned)        | GloVe 50D  | **~0.93**     |

---

## 🛠️ Requirements

Install the required packages using:

```bash
pip install pandas scikit-learn seaborn matplotlib tensorflow keras
```

Download GloVe embeddings (6B set) from:  
👉 https://nlp.stanford.edu/projects/glove/

Extract `glove.6B.50d.txt` and/or `glove.6B.100d.txt` into your project directory.

---

## 🚀 How to Run

1. Ensure `train.csv`, `test.csv`, and GloVe files are in the same directory.
2. Run the script:

```bash
python assignment_script.py
```

3. View printed classification reports and confusion matrices.

---

## 📚 References
- [GloVe Embeddings](https://nlp.stanford.edu/projects/glove/)
- [Keras CNN Text Classification Tutorial](https://realpython.com/python-keras-text-classification/)
- [TF Keras Text Tutorial](https://www.tensorflow.org/tutorials/keras/text_classification)
