"""spelling"""






from symspellpy import SymSpell, Verbosity

def correction_orthographique(phrase, langue='fr', distance_max=2):
    sym_spell = SymSpell(max_dictionary_edit_distance=distance_max, prefix_length=7)

    # Charger le dictionnaire français
    dictionnaire_path = f"chemin/vers/le/dictionnaire/{langue}.txt"
    sym_spell.load_dictionary(dictionnaire_path, term_index=0, count_index=1)

    # Corriger les mots mal orthographiés dans la phrase
    suggestions = sym_spell.lookup_compound(phrase, max_edit_distance=distance_max)

    # Reconstruire la phrase corrigée
    phrase_corrigee = suggestions[0].term if suggestions else phrase

    return phrase_corrigee
