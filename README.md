# Extraction, Recognition and Research in Medical Records

This code was developed to automate the management of medical records by performing text extraction, named entity recognition (NER), and enabling precise search within the documents.
![image](https://github.com/ZinebAissaoui/Projet_S2D/assets/150697197/793232e6-b839-4d6f-9142-f6036ff8d7de)


## Code Functionality

### 1. Text Extraction

The code retrieves medical records in PDF format and detects the presence of scanned images. For native PDFs, it uses the pdfplumber library from PyPDF2 to extract text data. For scanned PDFs, optical character recognition (OCR) is performed using the doctr library. The extracted medical records are then stored in .txt files for further manipulation.

### 2. Named Entities Recognition

For blood test files, the code uses the Quaero corpus to train a Conditional Random Fields (CRF) Tagger, which identifies relevant named entities. For other types of documents, regular expressions (Regex) are used to recognize named entities. Afterward, metadata is generated, and the JSON schema is defined to structure the data. The information is then stored in a JSON database.

### 3. Search Engine

The code vectorizes the text data using transformers, then creates an embedding file. It then calculates the similarity score using cosine similarity between the search query and the embeddings of the medical records. Finally, the code displays similar words and their context within the medical records.

## 4. Folders description in this Github

- **Analyse Folder**: Contains the code that uses the CRF Tagger to extract named entities and structure them for storage in JSON files. Other codes deal with processing the QUAERO database and training the CRF model.
- **Ordonnance Folder**:  For processing prescriptions, extracting named entities, and structuring them in JSON files.
- **FichePatient Folder**: For processing patient records and creating regular expressions (Regex).
- **CR Folder**: For processing radiology and scan reports, extracting named entities, and structuring them in JSON files.
- **Jsons**: SStores files in JSON format, including metadata files as well as embeddings.
- **Moteur de Recherche**: Development of all functionalities related to the search engine, including embeddings and similarities.
- **Fichier Total Run**: Groups all the functions used for data extraction, NER, creation of JSON files, as well as embedding the metadata and storing them.
- **P001**: Patient folder used for testing purposes.
