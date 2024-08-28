from sentence_transformers import SentenceTransformer, util
import json
from scipy.spatial.distance import cosine
import heapq
import numpy as np
embedder = SentenceTransformer('all-MiniLM-L6-v2')  ## embedding 

def mdr(file):
    # Ouvrir le fichier JSON
    with open(file, 'r') as json_file:
        data = json.load(json_file)

    # Supposons que input_encoded contienne le vecteur encodé de votre entrée
    mot_rechercher = input("Veuillez entrer le mot rechercher : ")

    input_encoded = np.array(embedder.encode(mot_rechercher, convert_to_tensor=True))

    # Valeurs initiales pour stocker les similarités cosinus les plus élevées
    top_similarities = []  # Utilisez un tas pour maintenir les 5 meilleures similarités

    # Parcourir les valeurs encodées
    for liste in data:
        for item in liste:
            if item['key']=='IdPatient':
                idpatient=item['value']
            if 'encoded_value' in item:
                encoded_value = np.array(item['encoded_value'])
                key = item['key']
                value = item['value']
                if 'sub_key' in item:
                    sub_key = item['sub_key']
                else:
                    sub_key = ""

                # Vérifier que les vecteurs sont unidimensionnels
                if input_encoded.ndim == 1 and encoded_value.ndim == 1:
                    # Calculer la similarité cosinus entre input_encoded et encoded_value
                    similarity = 1 - cosine(input_encoded, encoded_value)
                    try:
                        # Mettre à jour les 5 meilleures similarités
                        if len(top_similarities) < 5:
                            heapq.heappush(top_similarities, (similarity, value, key, sub_key,idpatient))
                        else:
                            heapq.heappushpop(top_similarities, (similarity, value, key, sub_key, idpatient))
                    except:
                        pass
                else:
                    print("Les dimensions des vecteurs ne sont pas correctes")

    # Tri des similarités par ordre décroissant
    top_similarities.sort(reverse=True)

    # Affichage des 5 meilleures similarités
    print("Les 5 valeurs encodées les plus proches sont :")
    for similarity, value, key, sub_key, idpatient in top_similarities:
        print("Clé :", key)
        print("Sub_key :", sub_key)
        print("Valeur :", value)
        print("Similarité cosinus :", similarity)
        print("ID :",idpatient)
        print()

mdr("C:/Users/zaiss/OneDrive/Documents/GitHub/Projet_S2D/JSONs/ConsultationEmbeddings.Json")