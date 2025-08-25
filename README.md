# Financial Document Fraud Detection AI

## Overview
AI system to extract structured data from financial documents (payslips, bank statements) and assess fraud risk using OCR + NLP + ML.

## Features
- OCR (Tesseract) text extraction
- NLP parsing & field normalization
- Feature engineering & fraud scoring
- Trust / risk scoring output

## Tech Stack
Python 3.11+
pytesseract, OpenCV
pandas, numpy, scikit-learn
spaCy (or NLTK)
(Planned) FastAPI for inference

## Prerequisites
1. Install Tesseract:
   - Windows (Chocolatey): `choco install tesseract`
   - Or download installer: https://github.com/tesseract-ocr/tesseract
2. (If using spaCy) `pip install spacy && python -m spacy download en_core_web_sm`
3. Python 3.11 recommended.

## Quick Start
```bash
git clone https://github.com/BA-3-2-Mavuti/Loan_Fraud_Detector.git
cd Loan_Fraud_Detector

python -m venv .venv
.\.venv\Scripts\activate          # Windows
pip install --upgrade pip
pip install -r requirements.txt

# Optional: verify tesseract path
tesseract --version
```

## Project Structure
```text
data/
  raw/            # original docs (PDF, images) - not committed
  processed/      # cleaned text / extracted fields
docs/
models/           # saved models / vectorizers
notebooks/
src/
  main_pipeline.py
  ocr/
  nlp/
  features/
  modeling/
  api/            # (future)
tests/
```

## Running the Pipeline
```bash
python src/main_pipeline.py --input data/raw --out data/processed
```

## Example (OCR -> Fraud Score)
```bash
python src/ocr/extract.py --file data/raw/sample_payslip.png --out temp/extracted.json
python src/modeling/predict.py --input temp/extracted.json --model models/latest.joblib
```

## Configuration
Planned central config file under `configs/` (e.g. YAML) for thresholds, feature flags.

## Team
- Mpho Matseka (Lead)
- Ntando Mbekwa
- Makhube Theoha
- Katleho Samuel Letsoho
- Pitso Nkotolane
- Dikeledi Madiboko
- Ayanda Ngamlana
- Zizipho Bulawa
- Palesa Mofokeng
- Zackaria Matshile Kgoale

## Data & Privacy
Do not commit real customer documents. Use redacted or synthetic samples.

## Roadmap
- [ ] Add FastAPI inference service
- [ ] Add model training script
- [ ] Add unit tests & CI
- [ ] Add drift monitoring

## Contributing
Create a feature branch, open PR with tests. Run `pytest` before submitting.

## License
Add a LICENSE file (MIT / Apache-2.0 recommended).
