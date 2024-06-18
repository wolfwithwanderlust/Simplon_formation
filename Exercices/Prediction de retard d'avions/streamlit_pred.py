import pandas as pd
import streamlit as st
import joblib

# Charger les données
data = pd.read_parquet('flights_data.parquet')

# Charger le modèle
model = joblib.load("model.joblib")

page_bg_img = '''
<style>
.stApp {
    background: url("https://nextvacay.com/wp-content/uploads/2022/08/best-day-of-the-week-to-book-a-flight.png");
    background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)

# Fonction pour la page de téléchargement CSV
def upload_csv_page():
    # Titre de la page
    st.title("Télécharger un fichier CSV")

    # Ajouter un composant de téléchargement de fichiers
    uploaded_file = st.file_uploader("Télécharger un fichier CSV", type=["csv"])

    # Vérifier si un fichier a été téléchargé
    if uploaded_file is not None:
        try:
            # Charger les données depuis le fichier CSV
            data = pd.read_csv(uploaded_file)

            # Afficher les données chargées
            st.write("Données téléchargées :")
            st.write(data)

            # Vérifier et convertir les types de colonnes si nécessaire
            expected_columns = {
                "DAY_OF_MONTH": int,
                "MONTH": int,
                "ORIGIN_CITY_NAME": str,
                "DEST_CITY_NAME": str,
                "UNIQUE_CARRIER": str,
                "DEP_TIME_BLK": str,
                "FL_NUM": int
            }

            for col, col_type in expected_columns.items():
                if col in data.columns:
                    try:
                        data[col] = data[col].astype(col_type)
                    except ValueError:
                        st.error(f"Impossible de convertir la colonne {col} en type {col_type}")
                        return
                else:
                    st.error(f"Colonne manquante dans le fichier CSV: {col}")
                    return

            # Faire la prédiction
            prediction = model.predict(data)

            # Ajouter une colonne "PREDICTION" au DataFrame avec les résultats de la prédiction
            data["PREDICTION"] = prediction

            # Afficher les résultats de la prédiction
            st.write("Résultats de la prédiction :")
            st.write(data)
        except Exception as e:
            st.error("Une erreur s'est produite lors du chargement des données : {}".format(str(e)))

# Fonction pour la page de formulaire de prédiction
def prediction_form_page():
    # Titre de la page
    st.title("Prédiction de retard des vols")

    # Listes déroulantes pour le jour, le mois, la ville de départ, la ville d'arrivée, la compagnie aérienne et le créneau horaire de départ
    days = sorted(data['DAY_OF_MONTH'].unique())
    months = sorted(data['MONTH'].unique())
    origin_cities = sorted(data['ORIGIN_CITY_NAME'].unique())
    dest_cities = sorted(data['DEST_CITY_NAME'].unique())
    carriers = sorted(data['UNIQUE_CARRIER'].unique())
    dep_times = sorted(data['DEP_TIME_BLK'].unique())

    # Afficher les listes déroulantes et les champs de texte
    day_of_month = st.selectbox("Jour du mois", days)
    month = st.selectbox("Mois", months)
    origin_city = st.selectbox("Ville de départ", origin_cities)
    dest_city = st.selectbox("Ville d'arrivée", dest_cities)
    unique_carrier = st.selectbox("Compagnie aérienne", carriers)
    dep_time_blk = st.selectbox("Heure de départ", dep_times)
    flight_number = st.text_input("Numéro de vol")

    # Ajouter un bouton pour lancer la prédiction
    if st.button("Prédire"):
        # Prétraiter les données
        input_data = {
            "DAY_OF_MONTH": [day_of_month],
            "MONTH": [month],
            "ORIGIN_CITY_NAME": [origin_city],
            "DEST_CITY_NAME": [dest_city],
            "UNIQUE_CARRIER": [unique_carrier],
            "DEP_TIME_BLK": [dep_time_blk],
            "FL_NUM": [flight_number]
        }

        # Convertir en DataFrame
        input_df = pd.DataFrame(input_data)

        # Faire la prédiction
        prediction = model.predict(input_df)

        # Afficher le résultat avec un peu de couleur
        if prediction[0] == 1:
            st.error("Le vol est prédit comme étant en retard")
        else:
            st.success("Le vol est prédit comme étant à l'heure")

# Titre principal
st.header("Application de Prédiction de Retard des Vols")

# Description
st.write("""
Cette application permet de prédire les retards des vols en fonction des données fournies.
Vous pouvez soit télécharger un fichier CSV contenant les informations des vols,
soit entrer les détails du vol via un formulaire pour obtenir une prédiction en temps réel.
""")

# Navigation entre les pages
page = st.sidebar.selectbox("Choisir une page", ["Télécharger un fichier CSV", "Formulaire de prédiction"])

# Afficher la page sélectionnée
if page == "Télécharger un fichier CSV":
    upload_csv_page()
elif page == "Formulaire de prédiction":
    prediction_form_page()
