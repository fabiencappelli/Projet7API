"""
streamlit run streamlit_app.py
"""
import streamlit as st
import requests
import json
import matplotlib.pyplot as plt
import os
from applicationinsights import TelemetryClient

# --- Application Insights TelemetryClient ---
conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")
if conn_str:
    tc = TelemetryClient(conn_str)
else:
    tc = None

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Text Classification Tester", layout="wide")

st.title("Interface de test pour l'API de classification de tweet")
st.write("Utilisez cette interface pour envoyer des requêtes à votre API FastAPI en local.")

st.markdown("---")
st.write("Entrez votre tweet ci-dessous (en anglais !)")

input_text = st.text_input(
    label="Votre tweet :", 
    value="This trip was terrible, I thought the plane would crash before even taking off"
)

if st.button("Envoyer la requête"):
    st.markdown("---")
    # Préparation du payload JSON
    payload = {"text": input_text}

    # Appel de l'API
    try:
        response = requests.post(API_URL + "/predict", json=payload)
        response.raise_for_status()
        results = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
    else:
        col1, col2 = st.columns([3, 1])

        with col1:
            st.subheader("Résultat")
            st.write("")
            if results["prediction"] == 0:
                st.error("Tweet négatif")
            else:
                st.success("Tweet positif")

            if st.button("⚠️ Mauvaise prédiction"):
                if tc:
                    tc.track_event(
                        "misprediction",
                        {
                            "tweet_text": input_text,
                            "predicted_label": results.get("prediction")
                        }
                    )
                    tc.flush()
                st.info("Merci pour votre retour !")

        with col2:
            st.subheader("Répartition des probabilités")
            left, center, right = st.columns([1,2,1])
            with center:
                p0, p1 = results["probabilities"]
                fig, ax = plt.subplots(figsize=(0.4, 0.4))
                fig.patch.set_alpha(0.0)
                ax.patch.set_alpha(0.0)

                ax.pie(
                    [p0, p1],
                    labels=["", ""],  # labels succincts
                    colors=["red", "green"],
                    startangle=90
                )
                ax.axis("equal")  # cercle parfait

                # Affiche en transparent
                st.pyplot(fig, transparent=True, use_container_width=False)