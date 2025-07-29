# Milvus-Rag-agentic-use-case

Ce repo présente un atelier (workshop) autour du Next-Gen Data Lakehouse pour la Génération d’IA (Generative AI) avec des technologies IBM Watsonx.ai, Milvus, Streamlit, etc. Le contexte est présenté lors de l’événement WAICF.

L’objectif est de démontrer :

L’orchestration d’agents intelligents (multi-agent)
L’intégration d’un Data Lakehouse (ingestion, stockage, vectorisation)
L’interfaçage via Streamlit pour afficher des résultats en direct (chat, RAG, streaming token par token).


# Structure du Projet

Ce projet est composé de plusieurs dossiers et fichiers clés, chacun ayant un rôle spécifique dans l’architecture et le fonctionnement de l’application.

## Contenu

- **Dockerfile**  
  Instructions pour construire l’image Docker de l’application (backend ou composants).

- **appout/**  
  Dossier de sortie éventuel pour les logs, les artefacts de build, etc.

- **conf/**  
  Contient les fichiers de configuration (YAML, JSON…) pour le backend ou l’ingestion de données.

- **docker-compose.yml**  
  Fichier d’orchestration multi-conteneurs : backend, front-end, Milvus, etc.

- **env/**  
  Fichiers d’environnement ou scripts d’activation.

- **front/**  
  Dossier dédié au front-end.

- **front_env/**  
  Environnement Python virtuel dédié à Streamlit & CrewAI.

- **Fichiers Streamlit (ex. app_agent.py, ...)**  
  Scripts de l’interface utilisateur.

- **openapi/**  
  Documentation et schémas des API REST.

- **ragapp.log**  
  Fichier principal de logs.

- **run_local_container.sh / run_local_standalone.sh**  
  Scripts pour le lancement local ou en conteneur.

- **src/**  
  Code source principal : backend, ingestion, etc.


  # Déploiement & Exécution

## Mode Docker / Compose (Serveur uniquement)

1. Installez [Docker](https://docs.docker.com/get-docker/) et [docker-compose](https://docs.docker.com/compose/install/).
2. À la racine du projet, lancez la commande suivante :
   ```sh
   docker-compose up --build
   ```
   > Seul le serveur est lancé pour l’instant.

---

## Pour l’interface utilisateur

1. Clonez le dépôt.
2. Installez ou activez l’environnement virtuel (par exemple, `env/front_env` selon la partie).

   ```sh
   # Créer et activer un environnement virtuel Python
   python3 -m venv env
   source env/bin/activate

   # Installer les dépendances à partir du fichier requirements.txt
   pip install -r requirements.txt
   ```

3. Lancez le backend (Flask, appels Watsonx.ai) via docker-compose si ce n’est pas déjà fait.
4. Lancez l’interface Streamlit depuis le dossier `front` :
   ```sh
   cd front
   streamlit run app_agent.py
   ```
5. Accédez à l’UI sur [http://localhost:8501](http://localhost:8501).
