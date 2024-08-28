
from Data_Extraction import extract_prescription

# Récupération des données à partir du fichier .txt 

output_file_path = 'Ordonnance\output.txt'
text = None

try:
    with open(output_file_path, 'r', encoding='utf-8') as output_file:
        text = output_file.read()
        print(text)
except FileNotFoundError:
    print(f"File '{output_file_path}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")


# Extraire la prescription
prescription= extract_prescription(text)

# Stockage des métadonnées
dictionnaire_data = {
    "Prescription": prescription
}
# Afficher le dictionnaire
print(dictionnaire_data)