# Extraction, Reconnaissance et Recherche dans les Dossiers Médicaux

Ce code a été élaboré dans le but d'automatiser la gestion des dossiers médicaux en effectuant l'extraction du texte, la reconnaissance des entités nommées, et en permettant une recherche précise au sein des documents.
![image](https://github.com/ZinebAissaoui/Projet_S2D/assets/150697197/793232e6-b839-4d6f-9142-f6036ff8d7de)


## Fonctionnement du Code

### 1. Extraction du Texte

Le code récupère les dossiers médicaux au format PDF et détecte la présence d'images scannées. Pour les PDF natifs, il utilise la bibliothèque pdfplumber de PyPDF2 pour extraire les données textuelles. Pour les PDF scannés, la reconnaissance optique de caractères (OCR) est effectuée à l'aide de la bibliothèque doctr. Les dossiers médicaux sont ensuite stockés dans des fichiers .txt pour une manipulation ultérieure.

### 2. Reconnaissance des Entités Nommées

Pour les fichiers d'analyses sanguines, le code utilise le corpus Quaero pour entraîner un Conditional Random Fields (CRF) Tagger qui identifie les entités nommées pertinentes. Pour les autres types de documents, des expressions régulières (Regex) sont employées pour reconnaître les entités nommées. Ensuite, les métadonnées sont générées, et le schéma JSON est défini pour structurer les données. Les informations sont ensuite stockées dans une base de données JSON.

### 3. Moteur de Recherche

Le code vectorise les données textuelles à l'aide de transformers, puis crée un fichier d'embedding. Ensuite, il calcule le score de similarité en utilisant la similarité cosinus entre la requête de recherche et les embeddings des dossiers médicaux. Enfin, le code affiche les mots similaires et leur contexte dans les dossiers médicaux.

## 4. Description des Dossiers de ce GitHub

- **Dossier Analyse**: Contient le code qui utilise le CRF Tagger pour extraire les entités nommées et les structurer pour les stocker dans des fichiers JSON. Les autres codes concernent le traitement de la base de données QUAERO et l'entraînement du modèle CRF.
- **Dossier Ordonnance**: Pour le traitement des ordonnances, l'extraction des entités nommées, et leur structuration dans des fichiers JSON.
- **Dossier FichePatient**: Pour le traitement des fiches patient et la création d'expressions régulières (Regex).
- **Dossier CR**: Pour le traitement des comptes rendus des radios et des scanners, l'extraction des entités nommées, et leur structuration dans des fichiers JSON.
- **Jsons**: Stocke les fichiers au format JSON, notamment ceux des métadonnées ainsi que ceux des embeddings.
- **Moteur de Recherche**: Développement de toutes les fonctionnalités liées au moteur de recherche, y compris les embeddings et les similarités.
- **Fichier Total Run**: Regroupe toutes les fonctions utilisées pour les parties d'extraction des données, NER, création des JSON, ainsi que l'embedding des métadonnées et leur stockage.
- **P001**: Dossier patient sur lequel nous avons effectué nos tests.
