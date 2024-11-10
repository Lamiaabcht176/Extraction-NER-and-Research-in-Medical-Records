
import pickle
import numpy as np
import pandas as pd
from sklearn_crfsuite import CRF
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib.pyplot as plt


# Utility functions
def save_pickle(data, file):
    """Saves data to a pickle file."""
    with open(file + ".pkl", "wb") as pick_file:
        pickle.dump(data, pick_file)


def load_pickle(file):
    """Loads data from a pickle file."""
    with open(file + ".pkl", "rb") as pick_file:
        data = pickle.load(pick_file)
    return data


def remplacer_virgule(chaine):
    """Replaces commas with dots in a string."""
    return chaine.replace(',', '.')


def est_nombre(s):
    """Checks if a string represents a number."""
    try:
        float(remplacer_virgule(s))
        return True
    except ValueError:
        return False


def find_first_number(lst):
    """Finds the first number in a list of strings."""
    for element in lst:
        if est_nombre(element):
            return float(remplacer_virgule(element))
    return None


# Functions related to CRF Model

def sent2labels(ltup):
    """Extracts labels from a list of tuples (token, label)."""
    return [label for (token, label) in ltup]


def sentences2labels(ls_tok_lab):
    """
    Converts a list of token-label tuples into a list of labels for each sentence.

    This function processes each sentence (a list of token-label tuples) using 
    the helper function `sent2labels`, which extracts the corresponding labels. 
    The resulting labels are stored in the list `y`.

    Args:
        sentences (list): A list of sentences, where each sentence is a list of 
                          (token, label) tuples.

    Returns:
        list: A list of labels for each sentence.
    """
    print("Extracting labels from sentences...")
    return [sent2labels(ltup) for ltup in ls_tok_lab]


def word2features(sent, i):
    """Generates feature dictionary for a word in a sentence."""
    token = sent[i][0]
    features = {
        'bias': 1.0,
        'word.lower()': token.lower(),
        'word.isupper()': token.isupper(),
        'word.istitle()': token.istitle(),
        'word.isdigit()': token.isdigit(),
        'word.isalnum()': token.isalnum(),
        'word.isalpha()': token.isalpha(),
        'word[-3:]': token[-3:],
        'word[-2:]': token[-2:],
    }
    if i > 0:
        token1 = sent[i - 1][0]
        features.update({
            '-1:word.lower()': token1.lower(),
            '-1:word.isupper()': token1.isupper(),
            '-1:word.istitle()': token1.istitle(),
            '-1:word.isdigit()': token1.isdigit(),
            '-1:word.isalnum()': token1.isalnum(),
            '-1:word.isalpha()': token1.isalpha(),
            '-1:word[-3:]': token1[-3:],
            '-1:word[-2:]': token1[-2:],
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        token1 = sent[i + 1][0]
        features.update({
            '+1:word.lower()': token1.lower(),
            '+1:word.istitle()': token1.istitle(),
            '+1:word.isupper()': token1.isupper(),
            '+1:word.isdigit()': token1.isdigit(),
            '+1:word.isalnum()': token1.isalnum(),
            '+1:word.isalpha()': token1.isalpha(),
            '+1:word[-3:]': token1[-3:],
            '+1:word[-2:]': token1[-2:],
        })
    else:
        features['EOS'] = True
    return features


def sentences2features(ls_tok_lab):
    """
    Converts sentences into a list of feature lists for CRF training.

    This function extracts features for each token in the input sentences using 
    `word2features`. Features include token case, capitalization, digit status, 
    alphanumeric/alphabetic status, last 2 and 3 characters, and features from 
    neighboring tokens. The function returns a list of feature sets for each sentence.

    Args:
        sentences (list): A list of sentences, where each sentence is a list of tokens.

    Returns:
        list: A list of lists of features for each sentence.
    """
    print("Converting sentences to features...")
    return [[word2features(ltup, i) for i in range(len(ltup))] for ltup in ls_tok_lab]


def train_test(ls_tok_lab):
    """
    Trains a CRF model for named entity recognition using 5-fold Cross Validation.

    This function processes input data (list of dictionaries with text and annotations), 
    extracting features and labels for each sentence. It trains a Conditional Random 
    Field (CRF) model and evaluates its performance using precision, recall, and F1-score. 
    The model is saved to a file named 'crf_tagger.pkl' for future use.

    Args:
        data (list): A list of dictionaries containing text and annotations.

    Returns:
        None
    """
    print("Begin training...")
    y = sentences2labels(ls_tok_lab)
    X = sentences2features(ls_tok_lab)

    c1, c2 = 0.005, 0.005
    all_possible_transitions = True
    max_iterations = 100
    algorithm = 'lbfgs'
    print("Model CRF algorithm=", algorithm, "c1=", c1, "c2=", c2,
          "max_iterations=", max_iterations, "all_possible_transitions=", all_possible_transitions)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    crf = CRF(algorithm=algorithm, c1=c1, c2=c2, max_iterations=max_iterations,
              all_possible_transitions=all_possible_transitions)

    print("Training CRF model...")
    crf.fit(X=X_train, y=y_train)

    # Make predictions
    pred = crf.predict(X_test)

    # Evaluate predictions
    mlb = MultiLabelBinarizer()
    y_test_bin = mlb.fit_transform(y_test)
    pred_bin = mlb.transform(pred)

    report = classification_report(y_true=y_test_bin, y_pred=pred_bin)
    print("Classification Report:")
    print(report)

    # Save model
    save_pickle(crf, "crf_tagger")

    return report


def step1(set):
    """Loads dataset, trains model, and returns report."""
    ls_tok_lab = load_pickle(set + '_txt_ann3')
    report = train_test(ls_tok_lab)
    return report


# Functions related to prediction and extraction

def predict_entities2(sentence, crf_model):
    """Predicts entities in a sentence using a trained CRF model."""
    tokens = [(token, '') for token in sentence.split()]
    features = [word2features(tokens, i) for i in range(len(tokens))]
    predicted_labels = crf_model.predict_single(features)
    return list(zip(sentence.split(), predicted_labels))


def process_sentence(sentence, crf_model):
    """Processes a sentence and identifies entities."""
    c = 0
    for e in sentence.split():
        if e.isalpha():
            k = predict_entities2(e, crf_model)
            for word, label in k:
                if label != 'NONE':
                    c += 1
    return c != 0


def extract_values_from_sentence(sentence):
    """Extracts values (numbers) from a sentence."""
    c, n = [], []
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


def correcte(v, min, max):
    """Checks if a value is within a specified range."""
    if min <= v <= max:
        return "Correcte"
    else:
        return "Pas correcte"


# Main execution
if __name__ == "__main__":
    set = 'train'  # Set dataset (e.g., 'train')
    report = step1(set)
    print(report)
    
    # Example for processing a sentence
    input_sentence = "Hémogiobine 13,5 g/di 8,0-14,0"
    crf_model = load_pickle("crf_tagger")
    result = process_sentence(input_sentence, crf_model)
    if result:
        c, n, min_val, max_val = extract_values_from_sentence(input_sentence)
        valeur = find_first_number(n)
        if len(c) > 1:
            print(valeur, min_val, max_val, " ".join(c[:-1]))
        else:
            print(valeur, min_val, max_val, " ".join(c))
