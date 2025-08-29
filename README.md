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

We have developed a proof-of-concept model using Python and core data science libraries. The key components of the solution are:

-   **Data Preprocessing:** Cleaning and preparing raw transaction data for modeling.
-   **Feature Engineering:** Creating new, informative features from the data (e.g., transaction frequency, time-based patterns).
-   **Handling Class Imbalance:** Using the SMOTE (Synthetic Minority Over-sampling Technique) to address the rarity of fraud examples in the dataset, ensuring the model learns effectively.
-   **Model Training:** Implementing and comparing several classification algorithms, including Logistic Regression (as a baseline) and Random Forest (as a more advanced model).
-   **Evaluation:** Assessing model performance using metrics appropriate for imbalanced datasets.

## 4. Tech Stack

-   **Language:** Python 3.x
-   **Core Libraries:**
    -   `pandas` & `numpy`: For data manipulation and numerical operations.
    -   `scikit-learn`: For building and evaluating machine learning models.
    -   `imbalanced-learn`: For handling class imbalance with SMOTE.
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
    -   Download it from the official source on Kaggle: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
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
-   Dikeledi Madiboko
-   Ayanda Ngamlana
-   Zizipho Bulawa
-   Palesa Mofokeng
-   Zackaria Matshile Kgoale
