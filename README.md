# Text Classifier App — DistilBERT Tweet Sentiment

Ce projet propose un **service complet de classification de tweets** (positif/négatif) basé sur DistilBERT.
Il s’appuie sur deux applications :

- **Une API FastAPI** pour exposer le modèle ML
- **Une interface utilisateur Streamlit** pour tester et explorer les résultats

Le tout est entièrement **dockerisé** et prêt à déployer !

---

## Sommaire

- [Aperçu du projet](#aperçu-du-projet)
- [Architecture](#architecture)
- [Installation rapide](#installation-rapide)
- [Configuration des secrets (`.env`)](#configuration-des-secrets-env)
- [Utilisation](#utilisation)
- [Déploiement avec Docker](#déploiement-avec-docker)
- [Structure des fichiers](#structure-des-fichiers)
- [Contribuer](#contribuer)
- [Auteurs et licence](#auteurs-et-licence)

---

## Aperçu du projet

- **Modèle utilisé** : DistilBERT, fine-tuné pour la classification binaire de tweets.
- **Fonctionnalités principales** :

  - Prédiction de sentiment via API REST (`/predict`)
  - Interface web interactive (Streamlit)
  - Feedback utilisateur loggué (Application Insights)
  - Monitoring et logs adaptés aux environnements pro

La création du projet est documentée [ici](https://www.fabiencappelli.com/projetoc7).

Vous pouvez consulter [l'API en ligne](https://projet7oc.fabiencappelli.com/).

---

## Architecture

- **API** : expose `/predict`, accepte un texte et retourne la prédiction + probabilités.
- **Streamlit** : envoie le texte, affiche le résultat, log les misprédictions.

---

## Installation rapide

### Prérequis

- Python 3.10+
- [Docker](https://www.docker.com/) (recommandé)
- [Git](https://git-scm.com/)

### Clonage

```bash
git clone https://github.com/fabiencappelli/Projet7API.git
cd Projet7API
```

---

## Configuration des secrets (`.env`)

Avant toute exécution, créez un fichier `.env` à la racine du projet.
Un exemple de structure est fourni dans `.env.example` :

```env
# .env.example

# Token pour accès aux données ou modèles sur DagsHub
DAGSHUB_TOKEN=

# Token Hugging Face (pour téléchargement de modèles privés ou API)
HF_TOKEN=

# Chaîne de connexion Application Insights (monitoring Azure)
APPLICATIONINSIGHTS_CONNECTION_STRING=
```

- **DAGSHUB_TOKEN** : Pour accéder aux artefacts ou modèles stockés sur [DagsHub](https://dagshub.com/).
- **HF_TOKEN** : Pour accéder à Hugging Face Hub ([créez votre token ici](https://huggingface.co/settings/tokens)) si besoin.
- **APPLICATIONINSIGHTS_CONNECTION_STRING** : Pour logger l’utilisation ou les feedbacks (optionnel sauf si monitoring activé).

---

## Utilisation

### Téléchargement du modèle

Avant la première utilisation, **il faut télécharger le modèle** (et le tokenizer) dans le dossier `deployed_model/`.
Ce dossier est créé automatiquement par le script suivant :

```bash
# Installer les dépendances nécessaires
pip install -r requirements.txt

# Télécharger le modèle (lance ce script une seule fois, secrets requis dans .env)
python download_model.py
```

### Lancement en local (hors Docker)

#### 1. Lancer l’API

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Lancer Streamlit

Dans un autre terminal :

```bash
streamlit run streamlit_app.py
```

---

## Déploiement avec Docker

**Méthode recommandée !**

Après avoir configuré votre `.env` et téléchargé le modèle (`python download_model.py`), lancez simplement :

```bash
docker-compose up --build
```

- API accessible sur [http://localhost:8000](http://localhost:8000)
- UI Streamlit sur [http://localhost:8501](http://localhost:8501)

**Variables d’environnement importantes (Streamlit) :**

- `API_URL` : URL de l’API backend (par défaut, `http://api:8000` en Docker)
- `APPLICATIONINSIGHTS_CONNECTION_STRING` : pour loguer les feedbacks utilisateur (Azure)

---

## Structure des fichiers

```
.
├── app.py                      # Backend FastAPI (API ML)
├── streamlit_app.py            # Frontend Streamlit
├── requirements-api.txt        # Dépendances pour la dockerisation de l’API
├── requirements-streamlit.txt  # Dépendances pour la dockerisation de Streamlit
├── requirements.txt            # Dépendances complètes (all-in-one)
├── docker-compose.yml          # Orchestration des services
├── Dockerfile.api              # Dockerfile pour l’API
├── Dockerfile.streamlit        # Dockerfile pour Streamlit
├── download_model.py           # Script de téléchargement du modèle
├── .env.example                # Exemples de variables d’environnement
├── deployed_model/             # Dossier attendu pour le modèle DistilBERT (créé par le script)
├── notebooks/                  # Dossier regroupant l'ensemble du projet exploratoire
└── tests/                      # Dossier de tests
```

---

## Contribuer

Les contributions sont bienvenues !

- Forkez le repo
- Créez une branche (`feature/ma-nouvelle-fonctionnalité`)
- Poussez vos modifs, ouvrez une **Pull Request**

---

## Auteurs et licence

Projet initial développé par \[Fabien Cappelli (Co Z/H)].
Licence [MIT](./LICENSE).

---
