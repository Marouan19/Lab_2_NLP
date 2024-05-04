# LAB 2

This project focuses on utilizing rule-based natural language processing (NLP) techniques and regular expressions (regex) to extract structured information from unstructured text data. The primary objective is to parse text containing product information, quantities, and prices, and then generate a formatted bill based on the extracted data.

## How to Clone the Code

To clone this repository, use the following command:

```bash
git clone git@github.com:Marouan19/Lab_2_NLP.git
```

## Part 1: Rule Based NLP and Regex:

In this part of the project, we employ various libraries and techniques to process textual data and extract relevant information:

### Imported Libraries and Packages

- **word2number**: This library is used to convert words representing numbers (e.g., "three") into numerical values.
- **tabulate**: Tabulate is utilized for creating formatted tables from lists of data.
- **re**: The re module provides support for regular expressions in Python, enabling us to define and apply complex patterns for text matching.
- **nltk.corpus.stopwords**: The NLTK library's stopwords corpus contains a list of common English stopwords, which are used to filter out irrelevant words.
- **spacy**: We leverage the spaCy library for tokenization and part-of-speech tagging.

### Steps

1. **Tokenization and Filtering**: The text is tokenized using spaCy, and tokens that are not adjectives are retained. Stopwords are removed from the tokenized text.
2. **Regex Pattern Matching**: A regex pattern is defined to extract product information, quantities, and prices from the filtered text.
3. **Processing Matches**: Matches obtained from applying the regex pattern are processed to extract relevant information, such as product names, quantities, unit prices, and total prices.
4. **Data Compilation**: The extracted information is compiled into structured data, typically in the form of lists or tuples.
5. **Output**: The final output consists of the parsed data, usually presented as a list of tuples containing product details, quantities, unit prices, and total prices.

### Example Result

The following example demonstrates the result of parsing a sample text:

```python
[('Samsung smartphones', 3, 150.0, 450.0),
 ('banana', 4, 1.2, 4.8),
 ('Hamburger', 1, 4.5, 4.5)]
```

This structured data can then be used to generate a bill or perform further analysis.

## Part 2: word Embedding:

## Bibliographies
- **word2number**: This library converts words representing numbers into their numerical counterparts, making it useful for extracting numerical quantities from text.
- **tabulate**: Used for pretty-printing tabular data in a visually appealing format, tabulate helps in presenting data in a structured manner.
- **nltk**: NLTK, or Natural Language Toolkit, is a widely-used Python library for natural language processing tasks like tokenization, stemming, lemmatization, and more.
- **spacy**: Another powerful NLP library, Spacy provides robust features for tasks such as tokenization, part-of-speech tagging, named entity recognition, and dependency parsing.
- **pyarabic**: Specifically designed for working with Arabic text, PyArabic offers functionality for Arabic text processing, including tokenization, stemming, and normalization.
- **string**: The string module provides a set of constants and classes for working with strings in Python, including a collection of punctuation characters.
- **sklearn**: Scikit-learn is a widely-used machine learning library offering various tools for data preprocessing, feature extraction, and model training.
- **matplotlib**: Matplotlib is a popular data visualization library in Python, used for creating plots, charts, and other graphical representations of data.
- **numpy**: NumPy is fundamental for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions.
- **gensim**: Gensim is a library for topic modeling and document similarity analysis, providing implementations of word embedding algorithms like Word2Vec and FastText.
- **tensorflow**: TensorFlow, developed by Google, is a deep learning framework for building and training neural networks. It offers high-level APIs for model construction and training.
- **qalsadi**: Qalsadi is an Arabic language toolkit that includes functionality for Arabic text lemmatization, stemming, and other linguistic operations.
- **tashaphyne**: Another library for Arabic text processing, Tashaphyne offers features like light stemming and normalization.
- **pymongo**: PyMongo is a Python library for interfacing with MongoDB databases, enabling Python applications to connect to and manipulate MongoDB databases conveniently.

## Explanation of Some Lines of Code

### Tokenization with Spacy

```python
word_tokens = nlp(text)
```
This line uses Spacy's NLP pipeline to tokenize the input text, splitting it into individual words.

### One-Hot Encoding

```python
one_hot_encoded_array = np.array(OneHotEncoder(tokens[0]))
```
Here, we use a custom function to perform one-hot encoding on the tokenized text, converting each word into a binary vector representation.

### Bag-of-Words (BoW) with CountVectorizer

```python
BOW = count_vect.fit_transform(cleaned_sent_tokens)
```
CountVectorizer is used to transform the cleaned sentence tokens into a Bag-of-Words representation, where each document (sentence) is represented by a vector indicating the count of each word in the vocabulary.

### TF-IDF Vectorization

```python
tfidf_matrix = tfidf.fit_transform(cleaned_sent_tokens)
```
TF-IDFVectorizer is employed to generate TF-IDF (Term Frequency-Inverse Document Frequency) vectors for the cleaned sentence tokens, capturing the importance of each word in the corpus.

### FastText Model Training
```python
model = FastText(sentences=[lemmatized_tokens], min_count=1)
```
FastText model is trained on the lemmatized tokens using the Gensim library. This model learns word embeddings based on subword information, allowing it to handle out-of-vocabulary words and morphological variations.

### t-SNE Visualization
```python
X_embedded = tsne.fit_transform(X)
```
t-SNE (t-Distributed Stochastic Neighbor Embedding) is applied to reduce the dimensionality of the data, allowing us to visualize high-dimensional word embeddings in a lower-dimensional space.

### TensorFlow Tokenizer
```python
tokenizer = Tokenizer()
tokenizer.fit_on_texts(tokens)
```
TensorFlow's Tokenizer is utilized to create a vocabulary and generate dense word embeddings based on pre-trained word vectors.

## Steps Followed

1. Import and preprocess text data.
2. Perform tokenization, removal of numbers, punctuation, and stop words.
3. Stem and lemmatize tokens to reduce dimensionality and improve performance.
4. Encode tokens using one-hot encoding technique.
5. Transform tokens into Bag-of-Words (BoW) representation using CountVectorizer.
6. Generate TF-IDF vectors to capture the importance of words in the corpus.
7. Train a FastText model to learn word embeddings from the preprocessed text data.
8. Visualize word embeddings using t-SNE to gain insights into word relationships.
9. Utilize TensorFlow's Tokenizer to create a vocabulary and generate dense word embeddings.

# Steps details

## Word Embedding

In this part of the project, we delve into word embedding techniques, focusing on creating word representations using different methods such as one-hot encoding and FastText.

1. **Import Data:** We first import data from a MongoDB collection, which contains text content. Then, we preprocess the text data by tokenizing, removing numbers, punctuation, and stop words, and finally, we stem and lemmatize the tokens.

2. **One-Hot Encoding:** We encode the tokens using one-hot encoding technique, creating a vocabulary and representing each word as a one-hot vector.

3. **Bag-of-Words (BoW):** We use CountVectorizer to transform the tokenized text into a matrix of token counts, representing each sentence as a Bag-of-Words representation.

4. **TF-IDF:** We utilize the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique to represent the tokenized text as numerical features based on their importance in the corpus.

5. **FastText:** We train a FastText model on the preprocessed text data to generate word embeddings. We then demonstrate how to retrieve word vectors and find similar words using the trained FastText model.

6. **t-SNE Visualization:** Finally, we visualize the word embeddings in 2D using t-SNE (t-Distributed Stochastic Neighbor Embedding) to gain insights into the relationships between words in the embedding space.

7. **TensorFlow Tokenizer:** We also showcase how to use TensorFlow's Tokenizer to create a vocabulary and generate dense word embeddings based on pre-trained word vectors.
