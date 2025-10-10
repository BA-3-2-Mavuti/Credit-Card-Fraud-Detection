import cv2
import pytesseract
import os

# Point pytesseract to your installation
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Path to image
image_path = "data/raw/Chase Transaction.JPG"

# Load the image
img = cv2.imread(image_path)

if img is None:
    print(f" Error: image not found or invalid path: {image_path}")
else:
    print(" Image loaded successfully.")

    # Preprocess image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Thresholding for better OCR
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Run OCR
    text = pytesseract.image_to_string(thresh)
    print("\n Extracted Text:")
    print(text)

    # Validate Financial Document
    def is_financial_document(text: str) -> bool:
        keywords = [
            "income", "employer", "salary", "pay", "bank",
            "balance", "account", "statement", "payslip"
        ]
        text_lower = text.lower()
        return any(word in text_lower for word in keywords)

    # Check and print validation result
    if is_financial_document(text):
        print("\n This is a financial document.")
    else:
        print("\n  This does not appear to be a financial document.")
