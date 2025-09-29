# AI-Powered Credit Card Fraud Detection
**Vaal University of Technology | AIBUY3A - Business Analysis 3.2**
---

## 1. Project Overview

This repository contains the work for the Business Analysis 3.2 project. The project proposes a real-time, AI-powered solution to detect fraudulent credit card transactions, addressing a critical challenge within the financial industry as part of the Fourth Industrial Revolution (4IR).

The system is designed as a supervised machine learning pipeline that ingests transaction data, performs feature engineering, and classifies each transaction as either legitimate or fraudulent, providing a probability score to aid human investigators.

## 2. Problem Statement

Credit card fraud is a major threat to financial institutions, leading to significant monetary losses and eroding customer trust. Traditional rule-based systems are often static and fail to adapt to the dynamic and sophisticated tactics used by fraudsters. They generate a high number of false positives (blocking legitimate transactions) and can miss novel fraud patterns, leading to both financial loss and poor customer experience.

Our solution aims to overcome these limitations by using a machine learning model that can learn from historical data to identify complex, non-obvious patterns indicative of fraud.

## 3. The AI Solution
-Load Data: We loaded the transactions dataset and checked for missing values and column types.

-Select Features: The id column was removed, and Class was used as the target (fraud or not fraud).

-Split Data: The dataset was split into training and testing sets to check the model on unseen data.

-Explore Data: We made a correlation heatmap and a cumulative fraud plot to understand patterns and trends in fraud cases.

-Baseline Model: We trained a Logistic Regression model as a first test, using feature scaling and balancing for rare fraud cases.

-Advanced Model: We trained a Random Forest model, which is better at finding complex patterns and detecting fraud.

-Check Performance: We used metrics like F1-score, accuracy, ROC-AUC, and confusion matrices to see how well the model works.

-Error Analysis: We looked at false positives (normal transactions flagged as fraud) and false negatives (fraud missed) to understand mistakes.

-Save Model: The trained model and scaler were saved, so we can use them later without retraining.

-Chatbot Deployment: We made a simple chatbot where users enter a Transaction ID and it tells them if it is frau or not fraud .

## 4. Tech Stack

-   **Language:** Python 3.x
-   **Core Libraries:**
    -   `pandas` & `numpy`: For data manipulation and numerical operations.
    -   `scikit-learn`: For building and evaluating machine learning models.
    -   `matplotlib` & `seaborn`: For data visualization and analysis.
    -   `jupyter`: For creating reproducible analysis notebooks.
    -   `joblib`: For saving and loading trained models.
-   **Project Management:** GitHub Projects

## 5. Repository Structure
```text
The repository is organized as follows to ensure clarity and reproducibility:
├── assets/                # Images and visual assets (e.g., social preview)
├── data/                  # Raw and processed datasets
├── models/                # Saved trained models (e.g., .joblib files)
├── notebooks/             # Jupyter notebooks for EDA, model training, and evaluation
├── reports/               # Final project report and poster
├── src/                   # Reusable Python scripts and helper functions
├── .gitignore             # Files and folders to ignore
├── README.md              # This file
└── requirements.txt       # Python dependencies for easy setup
```


## 6. Getting Started

To replicate the analysis and run the models, please follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/BA-3-2-Mavuti/Credit-Card-Fraud-Detection.git](https://github.com/BA-3-2-Mavuti/Credit-Card-Fraud-Detection.git)
    cd Credit-Card-Fraud-Detection
    ```
2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4. **Download the Dataset (IMPORTANT):**
    -   The dataset is too large to be included in this repository.
    -   Download it from the official source on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/nelgiriyewithana/credit-card-fraud-detection-dataset-2023/data).
    -   After downloading, place the `creditcard.csv` file inside the `data/` directory in this project.

5.  **Run the Jupyter Notebooks:**
    -   Launch Jupyter Lab:
        ```bash
        jupyter lab
        ```
    -   Open the `notebooks/` directory and run the notebooks in sequential order.

## 7. Team

-   Mpho Matseka (Lead)
-   Ntando Mbekwa
-   Makhube Theoha
-   Katleho Samuel Letsoho
-   Pitso Nkotolane
-   Dikeledi Madibogo
-   Ayanda Ngamlana
-   Zizipho Bulawa
-   Palesa Mofokeng
-   Zackaria Matshile Kgoale
