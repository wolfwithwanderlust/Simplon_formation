import streamlit as st
import joblib
import pandas as pd
import numpy as np
import math
from sklearn.metrics import mean_squared_error, r2_score

st.markdown(
    """
    <style>
    /* Appliquer un fond gris à tout le corps de l'application */
    [data-testid="stAppViewContainer"] {
        background-color: #e0e0e0; /* Gris clair */
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Création de la fonction pour calculer les distances de Haversine
def calculate_distances(df):
    # Coordonnées des points de référence
    la_lat, la_lon = 34.003342, -118.485832  # Los Angeles
    sf_lat, sf_lon = 37.787994, -122.407437  # San Francisco
    ontario_lat, ontario_lon = 34.068871, -117.651215  # Ontario

    # Fonction de calcul de la distance de Haversine
    def haversine_distance(lat1, lon1, lat2, lon2):
        R = 6371.0  # Rayon de la Terre en kilomètres
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad

        a = (math.sin(dlat / 2) ** 2 +
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        return R * c

    # Calculer les distances et les ajouter au DataFrame
    df['distance_LA'] = df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'], la_lat, la_lon), axis=1)
    df['distance_SF'] = df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'], sf_lat, sf_lon), axis=1)
    df['distance_Ontario'] = df.apply(lambda row: haversine_distance(row['latitude'], row['longitude'], ontario_lat, ontario_lon), axis=1)

    # Retourner les nouvelles colonnes de distances
    return df[['distance_LA', 'distance_SF', 'distance_Ontario']]


# Charger le modèle Joblib
model = joblib.load('code/wolfwithwanderlust/TP/Arturo/Prediction_immo/KNN_model.joblib')

# Ajouter une image et un en-tête
image_path = "code/wolfwithwanderlust/TP/Arturo/Prediction_immo/img/prediction_immo.jpg"
st.image(image_path, use_column_width=True)

# Titre de l'application
#st.title("Prédiction immobilière")

# Importer un fichier CSV
csv_file = st.file_uploader("Téléchargez un fichier CSV contenant les données des maisons", type=['csv'])

# Si un fichier CSV est téléchargé
if csv_file is not None:
    # Lire le fichier CSV
    data = pd.read_csv(csv_file)

    # Afficher un aperçu des données téléchargées
    st.write("Aperçu des données téléchargées :")
    st.dataframe(data.head())

    # Vérifier si les colonnes requises sont présentes dans le CSV
    required_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms',
                        'population', 'households', 'median_income', 'ocean_proximity', 'median_house_value']
    if all(col in data.columns for col in required_columns):
        # Ajouter un sous-titre pour signaler les prix prédits sur le CSV
        # Ajouter une image pour sous titre
        image_path = "code/wolfwithwanderlust/TP/Arturo/Prediction_immo/img/prediction_csv.jpg"
        st.image(image_path, use_column_width=True)
        #st.subheader("Prédiction des prix des maisons à partir du CSV")

        if st.button("Prédire les prix des maisons", key="predict_button"):
            # Sélectionner les colonnes nécessaires pour la prédiction
            X = data[required_columns[:-1]]
            y_true = data['median_house_value']

            # Faire les prédictions
            y_pred = model.predict(X)

            # Calculer R² et RMSE
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))

            # Afficher R² et RMSE en gras
            st.markdown("**<u>Scoring :</u>**", unsafe_allow_html=True)
            st.write(f"**R² : {r2:.4f}**", " - ", f"**RMSE : {rmse:.4f}**")


            # Ajouter les prédictions au dataframe
            data['predicted_price'] = y_pred

            # Afficher le dataframe avec les prédictions
            st.write("Données avec les prix prédits :")
            st.dataframe(data)

            # Bouton pour télécharger les données avec les prédictions
            st.download_button(
                label="Télécharger les données avec les prédictions",
                data=data.to_csv(index=False).encode('utf-8'),
                file_name='maisons_avec_predicted_prices.csv',
                mime='text/csv',
                key="download_button",
            )

        # Ajouter un sous-titre pour la prédiction d'une maison unique7
        # Ajouter une image pour sous titre
        image_path = "code/wolfwithwanderlust/TP/Arturo/Prediction_immo/img/prediction_house.jpg"
        st.image(image_path, use_column_width=True)
        #st.subheader("Prédiction du prix d'une maison unique")

        # Saisie personnalisée des caractéristiques de la maison
        st.write("Saisissez les caractéristiques personnalisées pour une maison :")

        # Création des champs pour entrer les caractéristiques de la maison
        longitude = st.number_input('Longitude', value=data['longitude'].mean())
        latitude = st.number_input('Latitude', value=data['latitude'].mean())
        housing_median_age = st.number_input('Âge médian du logement', value=data['housing_median_age'].mean())
        total_rooms = st.number_input('Nombre total de pièces', value=data['total_rooms'].mean())
        total_bedrooms = st.number_input('Nombre total de chambres', value=data['total_bedrooms'].mean())
        population = st.number_input('Population', value=data['population'].mean())
        households = st.number_input('Foyers', value=data['households'].mean())
        median_income = st.number_input('Revenu médian', value=data['median_income'].mean())

        # Menu déroulant pour choisir la proximité de l'océan
        ocean_proximity_options = ['NEAR BAY', 'INLAND', 'NEAR OCEAN', 'ISLAND', 'proximity']
        ocean_proximity = st.selectbox('Proximité de l\'océan', options=ocean_proximity_options)

        # Préparer les données pour la prédiction personnalisée
        input_data = pd.DataFrame({
            'longitude': [longitude],
            'latitude': [latitude],
            'housing_median_age': [housing_median_age],
            'total_rooms': [total_rooms],
            'total_bedrooms': [total_bedrooms],
            'population': [population],
            'households': [households],
            'median_income': [median_income],
            'ocean_proximity': [ocean_proximity]
        })

        if st.button("Prédire le prix de la maison"):
            # Prédiction personnalisée
            predicted_price = model.predict(input_data)

            # Afficher le prix prédit
            st.write(f"Le prix prédit de la maison est : ${predicted_price[0]:,.2f}")
