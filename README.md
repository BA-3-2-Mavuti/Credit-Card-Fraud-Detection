# Financial Document Fraud Detection AI

## 📌 Overview
This project is an AI-powered system that processes financial documents (e.g., payslips, bank statements) to extract important details and detect possible fraud.  
It uses:
- **OCR (Optical Character Recognition)** to read documents
- **NLP (Natural Language Processing)** to extract structured data
- **Machine Learning** to flag fraudulent applications and assign a trust score

## 🎯 Goals
1. Automate document processing in financial applications.
2. Detect fraudulent or falsified information in loan/credit applications.
3. Provide a fraud risk score for each processed application.

## 🛠 Tech Stack
- **Python 3.x**
- **Libraries**: pandas, numpy, scikit-learn, pytesseract, OpenCV, spaCy/NLTK
- **Tools**: GitHub, Google Colab / VS Code

## 📂 Project Structure
/data -> datasets (CSV, PDF, images)
/docs -> reports, planning documents
/models -> saved ML models
/notebooks -> Jupyter/Colab notebooks for testing
/src -> source code for OCR, NLP, ML, integration

## 👨‍💻 Team Members
- Mpho Matseka (Project Lead)
- Ntando Mbekwa
- Makhube Theoha
- Katleho Letsoho
- Pitso
- Dikeledi
- Ayanda

## 🚀 How to Run
1. Clone this repository:
   ```bash
   git clone https://github.com/<org-name>/<repo-name>.git

2. Navigate into the folder:

  ```
cd <repo-name>
```
3. Install dependencies:

```
pip install -r requirements.txt
```
4. Run the main pipeline:
```
python src/main_pipeline.py
