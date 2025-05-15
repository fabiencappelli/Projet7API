import os
import streamlit as st
import requests
import matplotlib.pyplot as plt
from applicationinsights import TelemetryClient

# --- Configuration Application Insights ---
conn_str = os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING", "")
instr_key = None
for part in conn_str.split(';'):
    if part.startswith("InstrumentationKey="):
        instr_key = part.split('=')[1]
        break
if not instr_key:
    raise RuntimeError("APPLICATIONINSIGHTS_CONNECTION_STRING invalide ou manquante.")
'''
La chaîne de connexion en entier pose en fait des problèmes de connexion
'''

# Initialisation du client Application Insights
tc = TelemetryClient(instr_key)
# Envoi d'un événement de démarrage
'''
J'ai dû mettre ce test en place car malgré une connexion qui ne devait avoir aucun souci je n'arrivais pas à envoyer mes alertes
de mauvaise prédiction. Je voulais donc voir si j'arrivais à communiquer avec Azure Insights en dehors de mon bouton
'''
tc.track_event('startup_test', {'stage': 'startup'})
tc.flush()

# --- Interface Streamlit ---
st.set_page_config(page_title="Test de classification de tweet", layout="wide")
st.header("Classification de tweet avec un modèle de type DistilBERT")
st.write("Attention : le premier démarrage peut prendre un peu de temps. Veuillez patienter pour l'évaluation de votre premier tweet.")
st.markdown("---")
st.write("Entrez votre tweet ci-dessous (en anglais !)")

# Initialisation de l'état pour garder les résultats
'''
Cette structure avec l'état pour garder les résultats a été indispensable pour pouvoir communiquer les mispredictions
'''

if 'results' not in st.session_state:
    st.session_state['results'] = None

# Saisie utilisateur (avec clé pour persistance)
input_text = st.text_input(
    label="Votre tweet :",
    value=st.session_state.get('input_text', "This trip was terrible, I thought the plane would crash before even taking off"),
    key='input_text'
)

# Envoi de la requête à l'API
if st.button("Envoyer la requête", key='send_request'):
    try:
        response = requests.post(
            f"{os.getenv('API_URL', 'http://localhost:8000')}/predict",
            json={"text": input_text}
        )
        response.raise_for_status()
        st.session_state['results'] = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")

# Affichage des résultats si disponibles
if st.session_state['results']:
    results = st.session_state['results']
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Résultat")
        
        if results.get('prediction') == 0:
            st.error("Tweet négatif")
        else:
            st.success("Tweet positif")
            

        # Bouton de feedback pour misprédiction
        if st.button("⚠️ Mauvaise prédiction", key='mispred_button'):
            with st.spinner("Envoi du log de misprédiction..."):
                tc.track_event(
                    'misprediction',
                    {
                        'tweet_text': input_text,
                        'predicted_label': results.get('prediction')
                    }
                )
                tc.flush()
            # je n'ai jamais vu le spinner en action, cela va trop vite
            # st.success("Log envoyé à Application Insights !")
            st.info("Merci pour votre retour !")

    with col2:
        st.subheader("Répartition des probabilités")
        p0, p1 = results.get('probabilities', [0, 0])
        fig, ax = plt.subplots(figsize=(0.4, 0.4))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        ax.pie([p0, p1], labels=["", ""], colors=["red", "green"], startangle=90)
        ax.axis("equal")
        st.pyplot(fig, transparent=True, use_container_width=False)