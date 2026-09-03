import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chargement du dataset (mis en cache pour ne pas recharger a chaque clic)
@st.cache_data
def charger_donnees():
    df = pd.read_csv("cosmetics_biopharma.csv")
    vectorizer = TfidfVectorizer(stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(df["Ingredients"])
    return df, tfidf_matrix

df, tfidf_matrix = charger_donnees()


def recommander_produits(categorie=None, cible_motcle=None, budget_max=None, produit_reference=None, top_n=5):
    resultats = df.copy()

    if categorie:
        resultats = resultats[resultats["Categorie"] == categorie]
    if cible_motcle:
        resultats = resultats[resultats["Cible"].str.contains(cible_motcle, case=False, na=False)]
    if budget_max:
        resultats = resultats[resultats["Prix"] <= budget_max]

    if resultats.empty:
        return pd.DataFrame()

    if produit_reference is not None:
        sim_scores = cosine_similarity(
            tfidf_matrix[produit_reference], tfidf_matrix[resultats.index]
        ).flatten()
        resultats = resultats.copy()
        resultats["similarite"] = sim_scores
        resultats = resultats.sort_values(by="similarite", ascending=False)
        resultats = resultats[resultats.index != produit_reference]
        return resultats[["Nom", "Marque", "Categorie", "Prix", "similarite"]].head(top_n)
    else:
        resultats = resultats.sort_values(by="Prix", ascending=True)
        return resultats[["Nom", "Marque", "Categorie", "Prix", "Cible"]].head(top_n)


# 2. INTERFACE
st.set_page_config(page_title="Recommandation Biopharma", page_icon="🧴")
st.title("🧴 Systeme de recommandation - Laboratoires Biopharma")
st.write("Renseigne le profil du client pour recevoir des recommandations personnalisees issues du catalogue Biopharma.")

# --- Barre laterale : filtres ---
st.sidebar.header("Profil du client")

categories = ["Toutes"] + sorted(df["Categorie"].unique().tolist())
categorie = st.sidebar.selectbox("Categorie de produit", categories)
categorie = None if categorie == "Toutes" else categorie

cible_motcle = st.sidebar.text_input(
    "Besoin / type de peau ou cheveu (mot-cle)",
    placeholder="ex: sensible, sec, tache, bebe, cuir chevelu..."
)
cible_motcle = cible_motcle if cible_motcle else None

prix_min = int(df["Prix"].min())
prix_max = int(df["Prix"].max())
budget_max = st.sidebar.slider("Budget maximum (FCFA)", min_value=prix_min, max_value=prix_max, value=prix_max)

top_n = st.sidebar.slider("Nombre de recommandations", min_value=3, max_value=10, value=5)

# --- Zone principale : resultats ---
st.subheader("🎯 Produits recommandes")

resultats = recommander_produits(
    categorie=categorie, cible_motcle=cible_motcle, budget_max=budget_max, top_n=top_n
)

if resultats.empty:
    st.warning("Aucun produit ne correspond a ces criteres.")
else:
    st.dataframe(resultats, use_container_width=True)

# --- Section : recherche de produits similaires ---
st.subheader("🔍 Trouver des produits similaires a un produit du catalogue")

nom_recherche = st.text_input("Tape le nom (ou une partie du nom) d'un produit")

if nom_recherche:
    produits_trouves = df[df["Nom"].str.contains(nom_recherche, case=False, na=False)]

    if produits_trouves.empty:
        st.warning("Aucun produit trouve avec ce nom.")
    else:
        choix = st.selectbox(
            "Selectionne le produit exact",
            produits_trouves["Nom"] + " — " + produits_trouves["Marque"]
        )
        index_produit = produits_trouves[
            (produits_trouves["Nom"] + " — " + produits_trouves["Marque"]) == choix
        ].index[0]

        st.write("Produits similaires (bases sur les ingredients) :")
        similaires = recommander_produits(produit_reference=index_produit, top_n=top_n)
        st.dataframe(similaires, use_container_width=True)

# --- Apercu du catalogue complet ---
with st.expander("📋 Voir le catalogue complet Biopharma"):
    st.dataframe(df[["Nom", "Marque", "Categorie", "Prix", "Cible"]], use_container_width=True)
