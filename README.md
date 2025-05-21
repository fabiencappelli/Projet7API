# Text Classifier App — DistilBERT Tweet Sentiment

Ce projet propose un **service complet de classification de tweets** (positif/négatif) basé sur DistilBERT.
Il s’appuie sur deux applications :

- **Une API FastAPI** pour exposer le modèle ML
- **Une interface utilisateur Streamlit** pour tester et explorer les résultats

Le tout est entièrement **dockerisé** et prêt à déployer !

---

## Sommaire

- [Aperçu du projet](#aperçu-du-projet)
- [Architecture](#architecture)
- [Installation rapide](#installation-rapide)
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

La création du projet est documentée [ici](https://www.fabiencappelli.com/projetoc7)

Vous pouvez consulter [l'API en ligne](https://projet7oc.fabiencappelli.com/).

---

## Architecture

```
┌─────────────┐     POST /predict     ┌──────────────┐
│ Utilisateur │ <───────────────────> │ Streamlit UI │
└─────────────┘                       └──────────────┘
                                              │
                                      HTTP    │
                                              ▼
                               ┌──────────────────┐
                               │ FastAPI Backend  │
                               └──────────────────┘
```

- **API** : expose `/predict`, accepte un texte et retourne la prédiction + probabilités.
- **Streamlit** : envoie le texte, affiche le résultat, log les misprédictions.

---

## Installation rapide

### Prérequis

- Python 3.10+
- [Docker](https://www.docker.com/) recommandé
- [Git](https://git-scm.com/)

### Clonage

```bash
git clone https://github.com/fabiencappelli/Projet7API.git
cd text-classifier-app
```

### Lancement en local (sans Docker)

#### 1. Lancer l’API

```bash
pip install -r requirements-api.txt
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

#### 2. Lancer Streamlit

Dans un autre terminal :

```bash
pip install -r requirements-streamlit.txt
streamlit run streamlit_app.py
```

#### (Optionnel) Installer toutes les dépendances (dev, test, API, UI)

```bash
pip install -r requirements.txt
```

---

## Déploiement avec Docker

**La méthode recommandée !**

Un simple :

```bash
docker-compose up --build
```

- API accessible sur [http://localhost:8000](http://localhost:8000)
- UI Streamlit sur [http://localhost:8501](http://localhost:8501)

**Variables d’environnement importantes (Streamlit) :**

- `API_URL` : URL de l’API backend (par défaut, `http://api:8000` en Docker)
- `APPLICATIONINSIGHTS_CONNECTION_STRING` : pour loguer les feedbacks utilisateur (Azure)

---

## Utilisation

### Interface utilisateur

- Ouvrir [http://localhost:8501](http://localhost:8501)
- Entrer un tweet (en anglais)
- Obtenir la prédiction de sentiment (positif/négatif)
- Possibilité de signaler une mauvaise prédiction (envoyée à Application Insights)

### API

- Documentation interactive : [http://localhost:8000/docs](http://localhost:8000/docs)
- Exemple d’appel :

```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text":"My tweet is awesome!"}'
```

---

## Structure des fichiers

```
.
├── app.py                  # Backend FastAPI (API ML)
├── streamlit_app.py        # Frontend Streamlit
├── requirements-api.txt    # Dépendances pour l’API
├── requirements-streamlit.txt # Dépendances pour Streamlit
├── requirements.txt        # Dépendances complètes (all-in-one)
├── docker-compose.yml      # Orchestration des services
├── Dockerfile.api          # Dockerfile pour l’API (non affiché ici)
├── Dockerfile.streamlit    # Dockerfile pour Streamlit (non affiché ici)
└── deployed_model/         # Dossier attendu pour le modèle DistilBERT
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
