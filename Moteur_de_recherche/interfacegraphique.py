import tkinter as tk
from tkinter import filedialog
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine
import heapq
import numpy as np
import json

class MDRApp:
    def __init__(self, master):
        self.master = master
        master.title("Moteur de Recherche - Similarités Cosinus")

        self.label = tk.Label(master, text="Sélectionnez le fichier JSON:")
        self.label.pack()
        self.json_file_path = "C:/Users/zaiss/OneDrive/Documents/GitHub/Projet_S2D/JSONs/ConsultationEmbeddings.Json"

        self.button = tk.Button(master, text="Parcourir", command=self.select_json_file)
        self.button.pack()


        self.search_entry_label = tk.Label(master, text="Mot à rechercher:")
        self.search_entry_label.pack()

        self.search_entry = tk.Entry(master)
        self.search_entry.pack()

        self.result_label = tk.Label(master, text="")
        self.result_label.pack()

        self.search_button = tk.Button(master, text="Rechercher", command=self.search)
        self.search_button.pack()

    def select_json_file(self):
        self.selected_file = filedialog.askopenfilename(title="Sélectionnez le fichier JSON", filetypes=[("Fichiers JSON", "*.json")])
        self.label.config(text=f"Fichier sélectionné : {self.selected_file}")

    def search(self):
        if not self.selected_file:
            self.result_label.config(text="Veuillez sélectionner un fichier JSON.")
            return

        mot_rechercher = self.search_entry.get()
        if not mot_rechercher:
            self.result_label.config(text="Veuillez entrer un mot à rechercher.")
            return

        embedder = SentenceTransformer('all-MiniLM-L6-v2')

        with open(self.selected_file, 'r') as json_file:
            data = json.load(json_file)

        input_encoded = np.array(embedder.encode(mot_rechercher, convert_to_tensor=True))

        top_similarities = []

        for liste in data:
            for item in liste:
                if item['key'] == 'IdPatient':
                    idpatient = item['value']
                if 'encoded_value' in item:
                    encoded_value = np.array(item['encoded_value'])
                    key = item['key']
                    value = item['value']
                    if 'sub_key' in item:
                        sub_key = item['sub_key']
                    else:
                        sub_key = ""

                    if input_encoded.ndim == 1 and encoded_value.ndim == 1:
                        similarity = 1 - cosine(input_encoded, encoded_value)
                        try:
                            if len(top_similarities) < 5:
                                heapq.heappush(top_similarities, (similarity, value, key, sub_key, idpatient))
                            else:
                                heapq.heappushpop(top_similarities, (similarity, value, key, sub_key, idpatient))
                        except:
                            pass
                    else:
                        print("Les dimensions des vecteurs ne sont pas correctes")

        top_similarities.sort(reverse=True)

        result_text = "Les 5 valeurs encodées les plus proches sont :\n\n"
        for similarity, value, key, sub_key, idpatient in top_similarities:
            result_text += f"Clé : {key}\n"
            result_text += f"Sub_key : {sub_key}\n"
            result_text += f"Valeur : {value}\n"
            result_text += f"Similarité cosinus : {similarity}\n"
            result_text += f"ID : {idpatient}\n\n"

        self.result_label.config(text=result_text)

if __name__ == "__main__":
    root = tk.Tk()
    app = MDRApp(root)
    root.mainloop()
