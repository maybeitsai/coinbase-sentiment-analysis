# Coinbase Review Sentiment Analysis

## Project Description

This project aims to analyze the sentiment (positive, negative, or neutral) expressed in user reviews of the Coinbase application on the Google Play Store. The analysis is performed using a comprehensive machine learning pipeline involving data scraping, text cleaning, sentiment labeling, feature extraction, training multiple classification models, evaluation, and selecting the best model for inference on new text data.

## Workflow

The project follows these 10 phases:

1.  **Setup & Library Imports:** Preparing the environment and importing necessary libraries.
2.  **Data Acquisition (Scraping):** Fetching raw review data from the Google Play Store.
3.  **Exploratory Data Analysis (EDA):** Analyzing the raw data to understand its characteristics.
4.  **Initial Cleaning:** Removing duplicates, null values, performing light text cleaning, and filtering based on review length.
5.  **Sentiment Labeling & Dataset Balancing:** Assigning sentiment labels (positive, negative, neutral) using VADER and ratings, then balancing the number of samples per class.
6.  **Deep Data Cleaning:** Performing intensive text cleaning (stopwords, lemmatization, etc.) on the labeled and balanced data.
7.  **Data Splitting:** Dividing the clean data into Training, Validation, and Test sets.
8.  **Feature Extraction:** Transforming text into numerical representations using:
    *   TF-IDF (for classic ML models).
    *   Tokenization, Padding, and FastText Embeddings (for Deep Learning models).
9.  **Model Definition & Training:** Training and experimenting with several models:
    *   Artificial Neural Network (ANN) - TensorFlow/Keras
    *   Convolutional Neural Network (CNN) - TensorFlow/Keras
    *   Random Forest (RF) - Scikit-learn
    *   XGBoost - Scikit-learn (using XGBoost native API)
    *   LightGBM - Scikit-learn
10. **Evaluation & Best Model Selection:** Evaluating all models on the test set and selecting the best performer based on validation and test accuracy.
11. **Inference:** Using the best model to predict sentiment on unseen test samples and new text inputs.

## Dataset

*   **Source:** User reviews for the Coinbase Android app (`com.coinbase.android`) from the Google Play Store (US, English).
*   **Acquisition Method:** Scraping using the `google-play-scraper` library.
*   **Data Files (`data/` folder):**
    *   `reviews_1_raw.csv`: Raw data obtained from scraping.
    *   `reviews_3_pembersihan_awal.csv`: Data after initial cleaning and length filtering (Q1-Q3 words).
    *   `reviews_4_pelabelan.csv`: Data after VADER sentiment labeling and class balancing (10k samples/class).
    *   `reviews_5_final_clean.csv`: Final, deeply cleaned data (stopwords, lemmatization) ready for modeling.

## Installation & Setup

1.  **Prerequisites:** Python 3.x
2.  **Clone Repository:**
    ```bash
    git clone https://github.com/maybeitsai/coinbase-sentiment-analysis.git
    cd coinbase-sentiment-analysis
    ```
3.  **Create Virtual Environment (Optional but recommended):**
    ```bash
    python -m venv venv
    # Activate (Windows)
    venv\Scripts\activate
    # Activate (Linux/macOS)
    source venv/bin/activate
    ```
4.  **Install Dependencies:**
    ```bash
    pip install google-play-scraper pandas numpy nltk scikit-learn xgboost lightgbm matplotlib seaborn tensorflow fasttext-wheel
    ```
    *(Note: `fasttext-wheel` might be needed if the standard `fasttext` installation causes issues. Alternatively, create a `requirements.txt` file and use `pip install -r requirements.txt`)*.
5.  **Download NLTK Resources:** Run the notebook; the setup code will automatically download required NLTK resources (vader_lexicon, stopwords, wordnet) if they are not found.
6.  **Download FastText Model:** Run the notebook; the code in Phase 7 will attempt to download the `cc.en.300.bin` model if it's not present in the project directory. **Warning:** This download is large (~6GB) and may take a significant amount of time.
7.  **(Optional) GPU Setup:** If you have a compatible NVIDIA GPU and wish to use TensorFlow with GPU acceleration, ensure you have the appropriate CUDA drivers and cuDNN installed.

## Usage

1.  Open and run the Jupyter notebook `sentiment-analysis.ipynb` using Jupyter Lab or Jupyter Notebook.
2.  Execute the cells sequentially from top to bottom.
    *   Ensure the CSV data files are saved/read from the correct `data/` folder.
    *   The scraping process (Phase 1) will only run if the `data/reviews_1_raw.csv` file is not found.
    *   Cleaning and labeling processes (Phases 3-5) will also be skipped if their respective output files already exist, speeding up re-executions.
3.  Model training and evaluation results will be displayed in Phase 9.
4.  Phase 10 demonstrates using the best model for inference on new data via the `predict_sentiment` function.

## Models Explored

*   **Deep Learning (with FastText Embeddings):**
    *   Artificial Neural Network (ANN)
    *   Convolutional Neural Network (CNN)
*   **Classic Machine Learning (with TF-IDF Features):**
    *   Random Forest (RF)
    *   XGBoost
    *   LightGBM

## Evaluation Results

Models were evaluated based on accuracy on the training, validation, and test sets. The target accuracy was >92% for both train and test (ideal) or a minimum of >85% for the test set.

| Model    | Train Acc | Val Acc | Test Acc |
|----------|-----------|---------|----------|
| LightGBM | 0.9467    | 0.9326  | 0.9263   |
| RF       | 0.9335    | 0.9263  | 0.9193   |
| XGBoost  | 0.9329    | 0.9250  | 0.9166   |
| CNN      | 0.9284    | 0.9196  | 0.9133   |
| ANN      | 0.8855    | 0.8880  | 0.8770   |

<br>

**Best Model Selected:** **LightGBM**
*   Best Validation Accuracy: 0.9326
*   Test Accuracy: 0.9263
*   Target Accuracy (Train > 92%, Test > 92%) **MET.**

## Author

*   **Name:** Harry Mardika