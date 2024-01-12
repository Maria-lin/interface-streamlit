
import streamlit as st
from PIL import Image
import base64

st.set_page_config(page_title='Projet Data Mining', layout='wide')

# Fonction pour définir une image de fond dans Streamlit


def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


# Ajout de l'image de fond
add_bg_from_local('data\huh.png')

# Titre du projet
st.title('	:bar_chart: Projet Data Mining')


with st.expander("Description du projet", expanded=True):
    st.subheader('Faculté d’Informatique, 2ème année Master Informatique')
    st.write('Département d’IASD, Système Informatique Intelligent')
    st.write('Module : Data Mining')

    st.markdown('**Projet: Partie 1**')
    st.markdown('''
    Exploitation des données et Extraction des règles d’associations. 
    Les données du monde réel sont généralement bruitées, d'un volume énorme et peuvent provenir de sources hétérogènes...
    ''')

    st.markdown('**Projet: Partie 2**')
    st.markdown('''
    Cette partie du projet vise à appliquer des techniques d'apprentissage supervisé et non supervisé pour analyser la fertilité des sols, en utilisant des algorithmes comme KNN, Decision Trees, Random Forest et k-means, intégrés dans une interface utilisateur interactive. ''')

    st.markdown('**Ce projet a été réalisé par :**')
    st.markdown('- Charef Mounir')
    st.markdown('- Hendel Lyna Maria')
