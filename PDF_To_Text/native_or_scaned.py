import cv2
import numpy as np
from PyPDF2 import PdfReader

def is_scanned_pdf(pdf_path, threshold=0.2):
    # Open the PDF file
    pdf_reader = PdfReader(pdf_path)

    # Iterate through each page of the PDF
    for page_num in range(len(pdf_reader.pages)):
        # Extract the text content of the page
        text_content = pdf_reader.pages[page_num].extract_text()

        # Convert the text content to a NumPy array
        img = np.array(list(map(ord, text_content)), dtype=np.uint8).reshape(-1, 1)

        # Calculate the mean of the gradient on the text content
        try:
            gradient_mean = cv2.Laplacian(img, cv2.CV_64F).var()
        except cv2.error:
            # If an error occurs (e.g., empty image), consider it as a scanned PDF
            return True

        # If the mean of the gradient is below the threshold, the page likely contains a scanned image
        if gradient_mean < threshold:
            return True

    return False

