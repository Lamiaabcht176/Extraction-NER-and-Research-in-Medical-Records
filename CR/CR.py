from spellchecker import SpellChecker
from unidecode import unidecode
import re 
def open_file_txt(chemin_du_fichier):
    with open(chemin_du_fichier, "r", encoding="UTF-8") as fichier_entree:
        # Lisez le contenu du fichier
        lignes = fichier_entree.readlines()
    return lignes

def corriger_orthographe(lignes):
    spell = SpellChecker(language='fr')

    # Correction orthographique ligne par ligne
    lignes_corrigees = []
    for ligne in lignes:
        mots = ligne.split()
        mots_corriges = [spell.correction(mot) if spell.correction(mot) is not None else mot for mot in mots]
        ligne_corrigee = ' '.join(mots_corriges)
        lignes_corrigees.append(ligne_corrigee)

    return lignes_corrigees

def nettoyer_ligne(ligne):
     # Enlever les espaces blancs inutiles à la fin de la ligne
    ligne_nettoyee = ligne.strip()
    
    # Enlever les caractères non alphanumériques
    ligne_nettoyee = re.sub(r'[^a-zA-Z0-9\s]', '', ligne_nettoyee)
    
    # Vérifier si la ligne contient plus d'un caractère
    if len(ligne_nettoyee) > 2:
        return ligne_nettoyee
    else:
        return None  # Retourner None pour indiquer que la ligne ne doit pas être incluse

def extraction_CR_main(chemin_du_fichier,caract_fin,caract_debut):
    lignes= open_file_txt(chemin_du_fichier)
    lignes_corrigees=corriger_orthographe(lignes)
    lignes_extraites=[]


    # Recherche de l'indice de la ligne contenant "compte rendu"
    indices_debut = [i for i, ligne in enumerate(lignes_corrigees) if unidecode(caract_debut).lower() in unidecode(ligne).lower()]
    indices_fin= [i for i, ligne in enumerate(lignes_corrigees) if unidecode(caract_fin).lower() in unidecode(ligne).lower()]
    # Vérification s'il y a au moins une ligne contenant "compte rendu"
    if indices_debut:
        # Prendre la première occurrence de "compte rendu" et ajouter 1 pour obtenir la ligne suivante
        indice_debut = indices_debut[0]

    else :
        indices_debut = [i for i, ligne in enumerate(lignes_corrigees) if unidecode('compte rendu').lower() in unidecode(ligne).lower() or unidecode('bilan').lower() in unidecode(ligne).lower()]
        
        if  indices_debut:
            indice_debut = indices_debut[0]
            print(f"indice de début : {caract_debut} non trouvé, nous allons chercher compte rendu ou bilan et les considérer comme caractère de début")
        else:
            indice_debut =0
            print(f"indice de début : {caract_debut} non trouvé, nous allons considére la première ligne du document comme titre du rapport")

    # Vérification s'il y a au moins une ligne le caract de fin"
    if indices_fin:
        indice_fin = indices_fin[0]

    else :
        indice_fin = -1
        print(f"indice de fin : {caract_fin} non trouvé, ceci peut impacter les résultats")

    # Extraction de la partie du compte rendu
    partie_du_compte_rendu = lignes_corrigees[indice_debut:indice_fin]

    # Affichage de la partie du compte rendu
    for ligne in partie_du_compte_rendu:
        ligne_netoye=nettoyer_ligne(ligne)
        if ligne_netoye is not None:
            lignes_extraites.append(ligne_netoye)
    
    dict_radio={}
    dict_radio["intituleRapprt"]=lignes_extraites[0]
    dict_radio["description"]= lignes_extraites[1:-1]
    dict_radio["DecisionRapport"]=lignes_extraites[-1]
    return dict_radio

