# Analyse Automatique de Documents Juridiques
---

## Lien de l'application

https://projetnlp-l3tsirsyeho2x87gzgupwb.streamlit.app/

---

## Présentation

Ce projet répond à la problématique suivante :
**Comment comprendre automatiquement un document juridique ?**

Il permet de :
- Classifier automatiquement un document juridique en 14 catégories
- Résumer automatiquement le contenu du document
- Détecter les clauses à risque dans le texte
- Calculer un score de risque basé sur l'analyse des clauses

Le modèle est capable de classifier 14 types de documents juridiques français issus du dataset data.gouv.fr :
conditions générales de vente, politique de confidentialité, contrat de travail, règlement intérieur, mentions légales, conditions d'utilisation, charte de cookies, politique RGPD, convention collective, contrat de prestation, accord de confidentialité (NDA), statuts d'entreprise, contrat de bail, et autres documents juridiques.

---


## Dataset

**Source** : [data.gouv.fr](https://www.data.gouv.fr) — france-collection-dataset

| Statistique | Valeur |
|---|---|
| Documents total | 10 438 |
| Fournisseurs | 14+ |
| Types de documents | 14 |
| Format | Fichiers Markdown (.md) |

---


## Performances du modèle
MétriqueValeurAccuracy globale:     94.49%
Macro avg F1-score:                 96.91%
Nombre de classes:                  14 types de documents
Confiance médiane:                  99.90%
Modèle de langue:                   CamemBERT-large
Classifieur:                        Régression Logistique

---


## Stack technique

- **Langage** : Python 3.11
- **NLP / Embeddings** : sentence-transformers, CamemBERT-large
- **Machine Learning** : scikit-learn (Logistic Regression, Random Forest)
- **Feature Engineering** : TF-IDF, Sentence Embeddings (1024 dimensions)
- **Déploiement** : Streamlit
- **Données** : data.gouv.fr (france-collection-dataset)
- **Format de sauvegarde** : Parquet, Pickle

---


## Dépendances principales
txtstreamlit
torch==2.9.0
torchvision
transformers>=4.40.0
huggingface_hub>=0.23.0
sentence-transformers>=3.0.0
scikit-learn
pandas
numpy
joblib
matplotlib
Pillow
pyarrow

---

## Architecture du pipeline
Documents juridiques (.md)
        │
        ▼
┌─────────────────────┐
│  Nettoyage texte    │  BeautifulSoup + Regex
│  (clean_dataset)    │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Embeddings         │  CamemBERT-large (1024 dim)
│  (embedding)        │  dangvantuan/sentence-camembert-large
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Classification     │  Régression Logistique
│  (classification)   │  Accuracy : 94.49%
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Évaluation         │  F1, ROC, Matrice de confusion
│  (evaluation)       │  + Extraction mots-clés TF-IDF
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Application Web    │  Streamlit (app.py)
│  + Déploiement      │  Streamlit Cloud
└─────────────────────┘

---

## Structure du projet
projetNLP/
│
└── france-collection-dataset/          # Dataset brut (data.gouv.fr)
│
│
├── app.py                          # Application Streamlit principale
├── requirements.txt                # Dépendances Python
├── README.md                       # Ce fichier
├── clean_dataset.ipynb             # Nettoyage et préparation du corpus juridique
├── embeddings.ipynb                # Génération des embeddings avec CamemBERT-large
├── classification.ipynb            # Entraînement et comparaison des modèles
├── evaluation.ipynb                # Évaluation complète et extraction des mots-clés TF-IDF
│
│
└── outputs/
    ├── modele_classification.pkl   # Modèle entraîné (Régression Logistique)
    ├── label_encoder.pkl           # Encodeur des 14 classes
    ├── tfidf_vectorizer.pkl        # Vectoriseur TF-IDF
    ├── keywords_juridiques.parquet # Mots-clés par type de document
    ├── eval_confusion_normalisee.png
    ├── eval_f1_par_classe.png
    ├── eval_roc_curves.png
    └── eval_confiance.png

---


## Types de documents classifiés

| Catégorie | Description |
|---|---|
| Privacy Policy | Politique de confidentialité |
| General Conditions of Sale | Conditions générales de vente |
| Terms of Service | Conditions d'utilisation |
| Trackers Policy | Politique des cookies |
| Community Guidelines | Règles de la communauté |
| Conditions of Carriage | Conditions de transport |
| Commercial Terms | Conditions commerciales |
| Acceptable Use Policy | Politique d'utilisation acceptable |
| Copyright Claims Policy | Politique de réclamations de droits |
| Seller Warranty | Garantie vendeur |
| Service Level Agreement | Accord de niveau de service |
| Imprint | Mentions légales |
| Business Privacy Policy | Politique de confidentialité professionnelle |

---



## Fonctionnalités de l'application

### Page 1 — Analyser un document
- Saisie de texte ou upload de fichier `.txt`
- Prédiction du type de document avec score de confiance
- Top 3 des classes les plus probables
- Extraction des mots-clés TF-IDF
- Résumé automatique extractif
- Détection des clauses à risque avec passages exacts
- Score de risque global (Faible / Modéré / Élevé)

exemple de texte à insérer:

Vous consentez à la collecte et au traitement de vos données personnelles dans le cadre de l’utilisation du service. Ces données pourront être conservées pour une durée indéterminée et utilisées à des fins d’analyse comportementale. Elles pourront également être transférées vers des pays ne disposant pas d’un niveau de protection adéquat, sans garantie supplémentaire.

### Page 2 — Tableau de bord
- Métriques globales du modèle
- Matrice de confusion normalisée
- Courbes ROC multi-classes
- Distribution des scores de confiance

### Page 3 — Mots-clés par classe
- Visualisation des mots-clés TF-IDF par type de document
- Graphique des scores et tableau complet

---

## Clauses à risque détectées

L'application analyse 4 catégories de clauses risquées :

| Catégorie | Exemples |
|---|---|
| Limitation de responsabilité | *non responsable, en aucun cas, sans garantie* |
| Cession de données | *partenaires commerciaux, à des fins publicitaires* |
| Résiliation abusive | *sans préavis, sans indemnité, à sa seule discrétion* |
| Engagement excessif | *irrévocablement, perpétuellement, licence mondiale* |

---



## Pipeline ML

```
Texte brut
    │
    ▼
Nettoyage (BeautifulSoup + regex)
    │
    ▼
Embeddings CamemBERT-large (1024 dimensions)
    │
    ▼
Logistic Regression
    │
    ▼
Classification (14 classes) + Confiance
```

---

## Auteur

ANDRIANASOLO LALA Teddy
Projet réalisé dans le cadre d'un examen NLP — Analyse automatique de documents juridiques (Sujet 14).

---

## Licence

Ce projet est à usage éducatif.