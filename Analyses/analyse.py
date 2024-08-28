

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


def remplacer_virgule(chaine):
    return chaine.replace(',', '.')

def process_sentence(sentence, crf_model):
    c = 0
    for e in sentence.split():
        if e.isalpha():
            k = predict_entities2(e, crf_model)
            for word, label in k:
                if label != 'NONE':
                    c += 1
                else:
                    pass
        else:
            pass
    return c != 0

def extract_values_from_sentence(sentence):
    c = []
    n = []
    for e in sentence.split():
        if e.isalpha():
            c.append(e)
        else:
            n.append(e)

    min_val, max_val = None, None

    for element in n:
        if '-' in element:
            valeur_avec_tiret = element
            valeurs = valeur_avec_tiret.split('-')
            min_val, max_val = float(remplacer_virgule(valeurs[0])), float(remplacer_virgule(valeurs[1]))
            break

    return c, n, min_val, max_val

def find_first_number(lst):
    for element in lst:
        if est_nombre(element):
            return float(remplacer_virgule(element))
    return None

def est_nombre(s):
    try:
        float(remplacer_virgule(s))
        return True
    except ValueError:
        return False

# Exemple d'utilisation
sentence = " Hémogiobine 13,5 g/di 8,0-14,0"
crf_model = crf_model
result = process_sentence(sentence, crf_model)

if result:
    c, n, min_val, max_val = extract_values_from_sentence(sentence)
    valeur = find_first_number(n)
    if len(c)>1:
        print(valeur, min_val, max_val, " ".join(c[:-1]))
    else:
        print(valeur, min_val, max_val, " ".join(c))


def correcte(v,min,max):
    if min <= v <= max:
        return("Correcte")
    else:
        return("pas correcte")


def dict_analyse1(outputtxt):
    with open(outputtxt, 'r') as file:
        # Lire chaque ligne du fichier
        for line in file:
            # Appliquer le traitement si la ligne n'est pas vide
            if line.strip():
                result = process_sentence(line, crf_model)

                if result:
                    try:
                        c, n, min_val, max_val = extract_values_from_sentence(line)
                        valeur = find_first_number(n)
                        if valeur:
                            if min_val:
                                if len(c)>1:
                                    print(f"intituleAnalyse: {' '.join(c[:-1])}, Valeur d'analyse: {valeur}, decisionanalyse:{correcte(valeur, min_val, max_val)}")
                                else:
                                    print(f"intituleAnalyse: {' '.join(c)}, Valeur d'analyse: {valeur}, decisionanalyse:{correcte(valeur, min_val, max_val)}")


                    except:
                        pass


def dict_analyse(outputtxt):
    # Création d'un dictionnaire pour stocker les valeurs
    values_dict = []

    with open(outputtxt, 'r') as file:
        # Lire chaque ligne du fichier
        for line in file:
            # Appliquer le traitement si la ligne n'est pas vide
            if line.strip():
                result = process_sentence(line, crf_model)

                if result:
                    try:
                        c, n, min_val, max_val = extract_values_from_sentence(line)
                        valeur = find_first_number(n)
                        if valeur:
                            if min_val:
                                if len(c)>1:
                                    values_dict.append({
                                        "intituleAnalyse": ' '.join(c[:-1]),
                                        "Valeur d'analyse": valeur,
                                        "decisionanalyse": correcte(valeur, min_val, max_val)
                                    })
                                    #print(f"intituleAnalyse: {' '.join(c[:-1])}, Valeur d'analyse: {valeur}, decisionanalyse:{correcte(valeur, min_val, max_val)}")
                                else:
                                    values_dict.append({
                                        "intituleAnalyse": ' '.join(c),
                                        "Valeur d'analyse": valeur,
                                        "decisionanalyse": correcte(valeur, min_val, max_val)
                                    })
                                    #print(f"intituleAnalyse: {' '.join(c)}, Valeur d'analyse: {valeur}, decisionanalyse:{correcte(valeur, min_val, max_val)}")


                    except Exception as e:
                        print(f"An error occurred: {e}")
                        pass
    return values_dict

    #print(values_dict)





                    
