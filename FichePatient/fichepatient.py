#!/usr/bin/env python
# coding: utf-8

# In[6]:


import re
import json
def extract_personal_info(texte, field_list):
    extracted_info = {}

    nom = ''
    prenom = ''
    date_naissance = ''

    i = 0
    while i < len(field_list):
        field = field_list[i]
        if field == 'Adresse' or field == 'Situation familiale':
            regex_pattern = fr'{field}\s*([\s\S]+?)(?={field_list[i + 1]}|$)'
            lines = re.findall(regex_pattern, texte)
            value = ' '.join(line.strip().replace("\n", " ") for line in lines)
        else:
       
            regex_pattern = fr'{field}\s*([^\n]*)'
            match = re.search(regex_pattern, texte)
            value = match.group(1).strip().replace("\n", " ") if match else None

        extracted_info[field] = value

        # Extracting values for id creation
        if field == 'Nom':
            nom = value
        elif field == 'Prénom':
            prenom = value
        elif field == 'Date de naissance':
            date_naissance = value

        i += 1

    # Adding 'id' field to the dictionary as the first element
    extracted_info = {'id': f"{nom}{prenom}{date_naissance}", **extracted_info}

    return extracted_info

# fields_to_extract = ['Nom', 'Prénom', 'Date de naissance', 'Lieu de naissance', 'Adresse', 'Téléphone', 'E-mail', 'Situation familiale', 'Profession', 'sociale','Allergies']
# output_file_path= "C:\\Users\\zaiss\\OneDrive\\Documents\\GitHub\\Projet_S2D\\P0001\Text\\exempleNA.txt"
# with open(output_file_path, 'r', encoding="windows-1252") as output_file:
#                 extracted_text = output_file.read()
# personal_info = extract_personal_info(extracted_text, fields_to_extract)

# json_output = json.dumps(personal_info, indent=2, ensure_ascii=False)

# print(json_output)




