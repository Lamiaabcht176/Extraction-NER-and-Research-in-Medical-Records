from sentence_transformers import SentenceTransformer, util
import torch
import pandas as pd
import json
from scipy.spatial.distance import cosine
import heapq
import numpy as np
import os
embedder = SentenceTransformer('all-MiniLM-L6-v2')  ## embedding 
def embedding(file):
##################################################################    
# Charger le fichier JSON de consultation
    with open(file, 'r') as f:
    #with open('data.json', 'r') as f:
        data = json.load(f)
#################################################################
#################################################################
    # Créer une liste pour stocker les valeurs à écrire dans le fichier JSON
    # Créer une liste pour stocker les valeurs à écrire dans le fichier JSON
    c = []

    # Parcourir chaque consultation
    for consultation in data['Consultations']:
        # Parcourir chaque clé et valeur de la consultation
        values_to_write=[]
        for key, value in consultation.items():
            if value is not None:  # Vérifiez si la valeur n'est pas None
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            for key2, value2 in item.items():
                                if not isinstance(value2, float) and value2 is not None:  # Vérifiez si la valeur2 n'est pas None
                                    encoded_value = embedder.encode(value2, convert_to_tensor=True)
                                    values_to_write.append({
                                        "key": key,
                                        "sub_key": key2,
                                        "value":value2,
                                        "encoded_value": encoded_value.tolist()  # Convertir le tenseur en une liste Python
                                    })
                        elif not isinstance(item, int) and item is not None:  # Vérifiez si l'item n'est pas None
                            encoded_value = embedder.encode(item, convert_to_tensor=True)
                            values_to_write.append({
                                "key": key,
                                "value":item,
                                "encoded_value": encoded_value.tolist()  # Convertir le tenseur en une liste Python
                            })
                else:
                    if not isinstance(value, int):
                        encoded_value = embedder.encode(value, convert_to_tensor=True)
                        values_to_write.append({
                            "key": key,
                            "value":value,
                            "encoded_value": encoded_value.tolist()  # Convertir le tenseur en une liste Python
                        })
        c.append(values_to_write)                    
    embedding_filepath = os.path.join("C:/Users/zaiss/OneDrive/Documents/GitHub/Projet_S2D/JSONs", "ConsultationEmbeddings.Json")
    # Écrire les valeurs dans un fichier JSON
    with open(embedding_filepath, 'w') as json_file:
        json.dump(c, json_file)
embedding('C:/Users/zaiss/OneDrive/Documents/GitHub/Projet_S2D/JSONs/Consultations.Json')