import pdfplumber


def extract_text_from_pdfplumber(pdf_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            num_pages = len(pdf.pages)
            text = ''

            for page_num in range(num_pages):
                page = pdf.pages[page_num]
                text += page.extract_text()

        return text
    except pdfplumber.PDFSyntaxError as e:
        print(f"Error reading PDF: {e}")
        return None


def save_text_to_file(text, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(text)
        print(f"Text saved to {output_file}")
    except Exception as e:
        print(f"Error saving text to file: {e}")


