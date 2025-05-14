import os
import logging
import streamlit as st
import requests
import matplotlib.pyplot as plt
from azure.monitor.opentelemetry import configure_azure_monitor

# 1. Récupération et affichage de la chaîne de connexion (debug)
env_conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if not env_conn_str:
    raise RuntimeError("La variable APPLICATIONINSIGHTS_CONNECTION_STRING n'est pas définie.")

# 2. Configuration OpenTelemetry Azure Monitor
configure_azure_monitor(
    logger_name="mon_espace_logger",
    connection_string=env_conn_str
)

# 3. Création du logger et ajout d'un handler Console pour debug local
def get_logger():
    logger = logging.getLogger("mon_espace_logger")
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        # Handler pour envoyer sur console (stdout)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)
    return logger

logger = get_logger()

# 4. Journal de debug pour l'UI
debug_msgs = []
# Affiche un aperçu de la connection string (premiers caractères)
debug_msgs.append(f"ConnString (début): {env_conn_str[:30]}...")
# Liste des handlers attachés et leurs types
handler_info = [type(h).__name__ for h in logger.handlers]
debug_msgs.append(f"Handlers attachés: {handler_info}")
# Test d'envoi de log de démarrage
try:
    logger.info(
        "startup_test",
        extra={"custom_dimensions": {"stage": "startup_test"}}
    )
    debug_msgs.append("Test de log initial envoyé (startup_test).")
except Exception as e:
    debug_msgs.append(f"Erreur test log initial : {e}")

# 5. Configuration de l'app Streamlit
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Text Classification Tester", layout="wide")
st.title("Interface de test pour l'API de classification de tweet")

# Affichage du journal de debug
with st.expander("Journal de debug du logger au démarrage", expanded=True):
    for msg in debug_msgs:
        st.write(msg)

st.write("Utilisez cette interface pour envoyer des requêtes à votre API FastAPI en local.")
st.markdown("---")
st.write("Entrez votre tweet ci-dessous (en anglais !)")

# Saisie utilisateur
input_text = st.text_input(
    label="Votre tweet :",
    value=st.session_state.get("input_text", "This trip was terrible, I thought the plane would crash before even taking off")
)

# Envoi de la requête à l'API
if st.button("Envoyer la requête", key="send_request"):
    try:
        response = requests.post(f"{API_URL}/predict", json={"text": input_text})
        response.raise_for_status()
        st.session_state["results"] = response.json()
        st.session_state["input_text"] = input_text
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

# Affichage des résultats si disponibles
after_results = []
if "results" in st.session_state:
    results = st.session_state["results"]
    input_text = st.session_state["input_text"]
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Résultat")
        if results.get("prediction") == 0:
            st.error("Tweet négatif")
        else:
            st.success("Tweet positif")

        # Bouton de feedback
        if st.button("⚠️ Mauvaise prédiction", key="mispred_button"):
            with st.spinner("Envoi du log de misprédiction..."):
                try:
                    logger.info(
                        "misprediction",
                        extra={
                            'custom_dimensions': {
                                'tweet_text': input_text,
                                'predicted_label': results.get("prediction")
                            }
                        }
                    )
                    st.success("Log envoyé à Application Insights !")
                except Exception as e:
                    st.error(f"Erreur lors de l'envoi du log : {e}")
            st.info("Merci pour votre retour !")

    with col2:
        st.subheader("Répartition des probabilités")
        p0, p1 = results.get("probabilities", [0, 0])
        fig, ax = plt.subplots(figsize=(0.4, 0.4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.pie([p0, p1], labels=["", ""], startangle=90)
        ax.axis("equal")
        st.pyplot(fig, transparent=True, use_container_width=False)
