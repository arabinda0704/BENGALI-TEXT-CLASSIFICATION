# Bengali Text Classification

This project focuses on classifying Bengali text documents into predefined labels using machine learning algorithms and natural language processing (NLP) techniques. The methodology combines Latent Semantic Indexing (LSI) and Word Embedding models to represent documents and employs a Support Vector Machine (SVM) classifier for classification.

---

## Table of Contents
- [Introduction](#introduction)
- [Latent Semantic Indexing (LSI)](#latent-semantic-indexing-lsi)
- [Word Embedding Based Model](#word-embedding-based-model)
- [Proposed Methodology](#proposed-methodology)
  - [Preprocessing](#preprocessing)
  - [Creating Term by Document Matrix](#creating-term-by-document-matrix)
  - [SVD for Dimensionality Reduction](#svd-for-dimensionality-reduction)
  - [Word Embedding Using FastText](#word-embedding-using-fasttext)
  - [Matrix Concatenation](#matrix-concatenation)
- [Classification](#classification)
- [Dependencies](#dependencies)
- [License](#license)

---

## Introduction
**Bengali Text Classification** is the process of automatically categorizing Bengali text documents into predefined classes. It involves:
- Preprocessing Bengali text to remove stopwords and perform stemming.
- Representing documents using advanced NLP techniques like LSI and Word Embeddings.
- Training a machine learning classifier to predict the document labels.

---

## Latent Semantic Indexing (LSI)
Latent Semantic Indexing is used to analyze and retrieve semantic relationships between words.  
Key steps:
1. **Term-Document Matrix**: Rows represent terms, columns represent documents.
2. **Singular Value Decomposition (SVD)**: Decomposes the matrix into three parts:  
   - **U**: Term-by-concept matrix.
   - **Σ**: Concept-by-concept diagonal matrix.
   - **V**: Concept-by-document matrix.

The reduced matrix allows dimensionality reduction while retaining semantic relationships.

---

## Word Embedding Based Model
Word embeddings represent words as dense numerical vectors in a high-dimensional space.  
Steps:
1. Generate document embeddings using pretrained **FastText** word vectors.
2. Average the word vectors for all meaningful words in each document.

The final document representation is a dense numerical vector.

---

## Proposed Methodology

### Preprocessing
Steps for preprocessing Bengali text:
1. **Remove Punctuation**: Replace punctuation marks with blank spaces.
2. **Tokenization**: Split sentences into words or tokens.
3. **Stopword Removal**: Remove common Bengali stopwords.
4. **Stemming**: Reduce words to their base forms by removing suffixes.

Example:  
Original:  
`"আমি আজ অফিসে যেতে পারবো না কারণ আমি বাইরে আছি।"`  
Processed:  
`["অফিস", "যেত", "পার", "কারণ", "বাইর"]`

---

### Creating Term by Document Matrix
The preprocessed data is transformed into a **TF-IDF (Term Frequency-Inverse Document Frequency)** vectorized matrix:
- **TF**: Measures the frequency of terms in a document.
- **IDF**: Measures the importance of a term across all documents.

---

### SVD for Dimensionality Reduction
Apply **Singular Value Decomposition (SVD)** on the Term by Document Matrix to reduce dimensionality:
- Output is a **document-by-concept matrix**.
- Captures latent semantic relationships in the data.

---

### Word Embedding Using FastText
1. Use a pretrained Bengali word vector model (**FastText**).
2. Compute the average word vector for each document.
3. Create a **document-by-term matrix** (size: `1756 x 300`).

---

### Matrix Concatenation
Combine the SVD and Word Embedding matrices via element-wise addition:
- Create a unified representation by adding the document-by-concept and document-by-term matrices.
- The final matrix (size: `1756 x 300`) captures both semantic and contextual information.

---

## Classification

### Support Vector Machine (SVM)
The **SVM classifier** is used for classifying documents:
1. Split the data into training and testing sets.
2. Train the SVM on the training set using:
   - Linear or non-linear kernels for flexibility.
   - Hyperparameter tuning (e.g., dimensions in LSI).
3. Evaluate performance using metrics like:
   - **Accuracy**
   - **Precision**
   - **Recall**
   - **F1-Score**

---

## Dependencies
To run this project, ensure you have the following libraries installed:
- Python (>= 3.8)
- NumPy
- Pandas
- Scikit-learn
- FastText
- Matplotlib

Install the dependencies using:
```bash
pip install numpy pandas scikit-learn matplotlib

