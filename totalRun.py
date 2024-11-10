## OCR -- PDF to text
import os
import json
from PDF_To_Text.pdf_to_text import Extraction
from PDF_To_Text.pdfnative_to_text import extract_text_from_pdfplumber,save_text_to_file
from CR.CR import extraction_CR_main
from Ordonnance.Data_Extraction import extract_prescription
from FichePatient.fichepatient import extract_personal_info
from Analyses.analysecopy  import analyse
from JsonsLoad.RemplirJson import patient_Json, consultations_Json
from PDF_To_Text.native_or_scaned import is_scanned_pdf
from Moteur_de_recherche.Embeddings import embedding
#extracteur=Extraction("C:\Users\lamia\Desktop\Extraction-NER-Recherche\P0001\PDF\CRradio.pdf") # Insérer le nom de pdf, avant fait glisser le document dans la zone fichier à gauche
#export_lignes_fichier = extracteur.export_lignes("C:\Users\lamia\Desktop\Extraction-NER-Recherche\P0001\Text\CRradio.txt")


# Remplacez "Extraction" par le nom de votre classe

# Répertoire d'entrée pour les fichiers PDF
input_directory = r"C:\\Users\\lamia\\Desktop\\Extraction-NER-Recherche\\P0001\\PDF\\"

# Répertoire de sortie pour les fichiers texte
output_directory = r"C:\\Users\\lamia\\Desktop\\Extraction-NER-Recherche\\P0001\\Text\\"
print("######### Etape 1: PDF to Text ################")

# Parcours de tous les fichiers dans le répertoire d'entrée
for filename in os.listdir(input_directory):
    if filename.endswith(".pdf"):
        # Création du chemin complet pour le fichier PDF
        pdf_filepath = os.path.join(input_directory, filename)
            
        # Création du nom de fichier pour le fichier texte
        txt_filename = os.path.splitext(filename)[0] + ".txt"
        txt_filepath = os.path.join(output_directory, txt_filename)
        # si le document est scanné
        if is_scanned_pdf(pdf_filepath):
        # Création de l'instance de la classe Extraction
            extracteur = Extraction(pdf_filepath)
            extracteur.export_lignes(txt_filepath)
        else:
             extracteur= extract_text_from_pdfplumber(pdf_filepath)
             save_text_to_file(extracteur,txt_filepath)

        # Exportation des lignes dans le fichier texte
        
print("######### PDF To Text ################")
print("######### Etape 2 : Extraction des données  ################")

for filename in os.listdir(output_directory):
    # Compte rendu traitement
    if "cr" in filename.lower():
        print("----------début traitement compte rendu -----------")
        chemin_du_fichier = os.path.join(output_directory, filename) 
        caract_debut= "compte rendu" #input("entrez le premier mot du titre du compte rendu:")
        caract_fin= "Résidence"  #input("entrez le premier mot du pied du compte rendu:")
        dict_CR=extraction_CR_main(chemin_du_fichier,caract_fin,caract_debut)
        print("----------Fin traitement compte rendu ---------------")
        print(dict_CR)
    #Fiche client
    if "fiche" in filename.lower():
        print("----------début fiche client ----------------")
        fields_to_extract = ['Nom', 'Prénom', 'Date de naissance', 'Lieu de naissance', 'Adresse', 'Téléphone', 'E-mail', 'Situation familiale', 'Profession', 'sociale','Allergies et intolérances']
        output_file_path = os.path.join(output_directory, filename) 
        with open(output_file_path, 'r', encoding="UTF-8") as output_file:
                        extracted_text = output_file.read()
        personal_info = extract_personal_info(extracted_text, fields_to_extract)
        print(personal_info)
        json_output = json.dumps(personal_info, indent=2, ensure_ascii=False)
        print("----------fin fiche client ----------------")
    if "ordo" in filename.lower():
        print("----------début traitement prescription ----------------")
        output_file_path=os.path.join(output_directory, filename) 
        text = None
        try:
            with open(output_file_path, 'r', encoding="UTF-8") as output_file:
                text = output_file.read()
            
        except FileNotFoundError:
            print(f"File '{output_file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


        # Extraire la prescription
        prescription= extract_prescription(text)

        
        print(prescription)
        print("----------Fin traitement prescription ---------------")
    if "analyse" in filename.lower():
        print("----------début traitement Analyse ----------------")
        output_file_path=os.path.join(output_directory, filename) 
        text = None
        try:
            with open(output_file_path, 'r', encoding="UTF-8") as output_file:
                text = output_file.read()
                
        except FileNotFoundError:
            print(f"File '{output_file_path}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        listes_analyses=analyse(output_file_path)
        print(listes_analyses)
        print("----------Fin traitement analyse ---------------")
print("######### Etape 3 : Ajout au fichier JSON ################")
patient_Json_path= "C:\Users\lamia\Desktop\Extraction-NER-Recherche\JSONs\Patient.Json"
consultations_Json_path="C:\Users\lamia\Desktop\Extraction-NER-Recherche\JSONs\Consultations.Json"
idpatient=personal_info['id']
idmedecin='testMedecin01'
date='07/02/2024'

consultations_Json(consultations_Json_path,listes_analyses,dict_CR,idpatient,idmedecin,date,prescription)

patient_Json(patient_Json_path, personal_info)

print("######### Etape 4 : Embeddings ################")
path_consultation="C:\Users\lamia\Desktop\Extraction-NER-Recherche\JSONs\Consultations.Json"
embedding(path_consultation)

