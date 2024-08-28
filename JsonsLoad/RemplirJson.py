#Fiche Patient
import json
def patient_Json(patient_Json_path, dictionnaire_patient):
    newpatient = True
    with open(patient_Json_path,"r",encoding='UTF-8') as f:
            data_patient = json.load(f)
           
    patients=data_patient["patients"] # listes des dictionnaires patients
    
    for index, patient in enumerate(patients):
        
         #Update du dictionnaire patient
        if patient["id"]==dictionnaire_patient["id"]:
            
            newpatient=False
            patients[index]=dictionnaire_patient
            data_patient["patients"]=patients
            print(data_patient)
    
            break
    if newpatient :
        patients.append(dictionnaire_patient)
        data_patient["patients"]=patients
    with open(patient_Json_path, "w",  encoding='utf-8') as data_file:
        json.dump(data_patient, data_file, indent=2, ensure_ascii=False)

def consultations_Json(consultations_Json_path,list_analyse,dict_CR,idpatient,idmedecin,date,list_prescription=None,symptome=None,diagnostic=None):
    newconsul=True
    with open(consultations_Json_path,"r",encoding='UTF-8') as f:
            data_consultations = json.load(f)
    consultations=data_consultations["Consultations"] # listes des dictionnaires consultations
    idconsultation=idpatient+idmedecin+date
    for indx,consultation in enumerate(consultations):
        if idconsultation==consultation["IdConsultation"]:
            newconsul=False
            consultation["Prescription"]=list_prescription
            consultation["Analyses"]=list_analyse
            consultation["Rapports"]=[dict_CR]
            consultations[indx]=consultation
            data_consultations["Consultations"]=consultations
            
            break
    if newconsul: 
        consultations.append({
                                "IdConsultation" : idconsultation,
                                "IdPatient": idpatient,
                                "IdMedecin": idmedecin,    
                                "Date": date,
                                "Syptomes" : symptome,
                                "Diagnostic" : diagnostic,
                                "Prescription" : list_prescription,
                                "Nombrededocuments": 3,
                                "Analyses":list_analyse,
                                "Rapports":[
                                    dict_CR
                                    ]

                            })
        data_consultations["Consultations"]=consultations
    with open(consultations_Json_path, "w",  encoding='utf-8') as data_file:
        json.dump(data_consultations, data_file, indent=2, ensure_ascii=False)
             
             


            
         