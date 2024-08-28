

def word2features(sent, i):
    token = sent[i][0]     # take word i from the sentence
    features = {
        # setup the features
        'bias': 1.0,
        'word.lower()': token.lower(),  	# Is the token lowercase?
        'word.isupper()': token.isupper(),  # Is the token uppercase?
        'word.istitle()': token.istitle(),  # Does the token begin with a capital?
        'word.isdigit()': token.isdigit(),  # Is the token made up of digits?
        'word.isalnum()': token.isalnum(),  # Is the token formed by alphanumeric chars?
        'word.isalpha()': token.isalpha(),  # Is the token formed by alphabetic chars?
        'word[-3:]': token[-3:],		    # the last 3 chars of the token
        'word[-2:]': token[-2:], 	        # the last 2 chars of the token
    }
    if i > 0:  # if it is not the first word
        token1 = sent[i - 1][0]        # take the previous token
        features.update({               # update the features
            '-1:word.lower()': token1.lower(),      # Is the previous token lowercase?
            '-1:word.isupper()': token1.isupper(),  # Is the previous token uppercase?
            '-1:word.istitle()': token1.istitle(),  # Does it begin with a capital?
            '-1:word.isdigit()': token1.isdigit(),  # Is the previous token made up of digits?
            '-1:word.isalnum()': token1.isalnum(),  # Is the previous token formed by alphanumeric chars?
            '-1:word.isalpha()': token1.isalpha(),  # Is the previous token formed by alphabetic chars?
            '-1:word[-3:]': token1[-3:],            # the last 3 chars of the previous token
            '-1:word[-2:]': token1[-2:],            # the last 2 chars of the previous token
        })
    else:       # if it is the first word
        features['BOS'] = True  # set 'Begin Of Sentence'
    if i < len(sent) - 1:           # if it is not the last word
        token1 = sent[i + 1][0]     # take the next word
        features.update({           # update the features:
            '+1:word.lower()': token1.lower(),      # Is the next token lowercase?
            '+1:word.istitle()': token1.istitle(),  # Does it begin with a capital?
            '+1:word.isupper()': token1.isupper(),  # Is the it uppercase?
            '+1:word.isdigit()': token1.isdigit(),  # Is the next token made up of digits?
            '+1:word.isalnum()': token1.isalnum(),  # Is the next token formed by alphanumeric chars?
            '+1:word.isalpha()': token1.isalpha(),  # Is the next token formed by alphabetic chars?
            '+1:word[-3:]': token1[-3:],            # the last 3 chars of the next token
            '+1:word[-2:]': token1[-2:],            # the last 2 chars of the next token
        })
    else:       # if it is the last word
        features['EOS'] = True  # set 'End Of Sentence'
    return features
    
def sentences2features(ls_tok_lab):
    print("converting sentences to features...")
    lfeat = []
    for ltup in ls_tok_lab:
        lfeat.append([word2features(ltup, i) for i in range(len(ltup))])
    return lfeat


import os
import pickle

def load_pickle(file):
    pick_file = open("Analyses\\"+file+".pkl", "rb")
    data = pickle.load(pick_file)
    pick_file.close()
    return data


def predict_entities2(sentence, crf_model):
    # Tokenize the input sentence
    tokens = [(token, '') for token in sentence.split()]  # assuming initial labels are empty

    # Extract features for each token
    features = [word2features(tokens, i) for i in range(len(tokens))]

    # Make predictions using the trained CRF model
    predicted_labels = crf_model.predict_single(features)

    # Combine tokens and predicted labels
    predicted_entities = list(zip(sentence.split(), predicted_labels))

    return predicted_entities

# Load the trained CRF model
crf_model = load_pickle("crf_tagger")

#######################################
#################################################Fonction pour pretraitement de mot avant de detecter label
def pretraitement(mot):
    mot=mot.replace(".", "")
    mot=mot.replace("-", "")
    mot=mot.replace(")", "")
    mot=mot.replace("(", "")
    return mot
#######################################
#################################################Supp les ,
def remplacer_virgule(chaine):
    return chaine.replace(',', '.')
#######################################
#################################################Supp les %
def pretraitement_digitz(mot):
    return mot.replace('%','')
#######################################
#################################################Verifier si c'est nombre
def is_digit(word):# Fonction qui verifie si c'est un nombre
    try:
        float(word)
        return True
    except ValueError:
        return False
#######################################
#################################################fonction qui verifie si la valeur d'analyse est correcte
def correcte(valeur, min_vals, max_vals):
    min_val = float(min_vals[0])
    max_val = float(max_vals[0])
    v = float(valeur[0])
    if min_val <= v <= max_val:
        return "Correcte"
    else:
        return "Pas correcte"
#######################################
#################################################Fonction finale
def detect_analyse(ligne):
    word_list = ligne.split()
    analyse = []
    values = []
    minn=[]
    maxx=[]
    result = {}
    #### Trouver les valeurs de l'analyse et min et max
    for e in range(0, len(word_list)):
        if ('(' in word_list[e]) and (')' in word_list[e]):
            valminmax=word_list[e].replace(")", "")
            valminmax=valminmax.replace("(", "")
            valeur_avec_tiret = valminmax
            valeurs = valeur_avec_tiret.split('-')
            if  (len(valeurs)>1):
                min_valu=remplacer_virgule(valeurs[0])
                max_valu=remplacer_virgule(valeurs[1])
                min_val, max_val = float(min_valu), float(max_valu)
                minn.append(min_val)
                maxx.append(max_val)   
        if is_digit(remplacer_virgule(word_list[e])):
            values.append(remplacer_virgule(word_list[e]))
            #values.append(word_list[e])
        else:
            mot = pretraitement(word_list[e])
            if mot:
                mot = mot.lower()
                if (predict_entities2(mot, crf_model))[0][1] != 'NONE':
                    analyse.append(mot)
    
    # Créer un dictionnaire si les deux listes ne sont pas vides
    if values and analyse  and maxx:
        result["Valeur d'analyse"] = values
        result['intituleAnalyse'] = analyse
        #result['min_val']=minn
        #result['max_val']=maxx
        result['decisionanalyse']=correcte(values,minn,maxx)
        return result
    else:
        return None

# Créer une liste pour stocker les dictionnaires
def analyse(fichier):
    dicts = []
    with open(fichier, 'r') as file:
        # Lire chaque ligne du fichier
        for line in file:
            dict_line = detect_analyse(line)
            if dict_line is not None:
                dicts.append(dict_line)
    return dicts


