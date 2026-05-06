import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from sentence_transformers import SentenceTransformer

## OUTPUT_DIR = r"E:\INSI\NLP\projet_exam\outputs"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")

st.set_page_config(
    page_title="Analyse Juridique IA",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title   { font-size:2.2rem; font-weight:700; color:#1E3A5F; }
    .subtitle     { font-size:1rem; color:#6B7280; margin-bottom:1.5rem; }
    .metric-box   { background:#F0F4FF; border-radius:10px; padding:1rem;
                    text-align:center; border-left:4px solid #2563EB; }
    .metric-val   { font-size:1.8rem; font-weight:700; color:#2563EB; }
    .metric-label { font-size:0.85rem; color:#6B7280; }
    .tag          { display:inline-block; background:#E0EDFF; color:#1D4ED8;
                    border-radius:20px; padding:3px 12px; margin:3px;
                    font-size:0.82rem; font-weight:500; }
    .warning-box  { background:#fcaf49; border-left:4px solid #F59E0B;
                    padding:0.8rem; border-radius:6px; }
    .success-box  { background:#D1FAE5; border-left:4px solid #10B981;
                    padding:0.8rem; border-radius:6px; }
</style>
""", unsafe_allow_html=True)

# CHARGEMENT DES MODÈLES

@st.cache_resource
def charger_modeles():
    clf   = joblib.load(f"{OUTPUT_DIR}/modele_classification.pkl")
    le    = joblib.load(f"{OUTPUT_DIR}/label_encoder.pkl")
    tfidf = joblib.load(f"{OUTPUT_DIR}/tfidf_vectorizer.pkl")
    model = SentenceTransformer("dangvantuan/sentence-camembert-large")
    return clf, le, tfidf, model

@st.cache_data
def charger_keywords():
    return pd.read_parquet(f"{OUTPUT_DIR}/keywords_juridiques.parquet")

# FONCTION D'ANALYSE

def analyser_document(texte, clf, le, tfidf, model):
    vec      = model.encode([texte[:512]], convert_to_numpy=True)
    label_id = clf.predict(vec)[0]
    probas   = clf.predict_proba(vec)[0]
    type_doc = le.inverse_transform([label_id])[0]

    top_3 = sorted(zip(le.classes_, probas), key=lambda x: x[1], reverse=True)[:3]

    feature_names = np.array(tfidf.get_feature_names_out())
    vec_tfidf     = tfidf.transform([texte])
    scores        = np.asarray(vec_tfidf.todense()).flatten()
    top_idx       = scores.argsort()[::-1][:12]
    mots_cles     = [feature_names[i] for i in top_idx if scores[i] > 0]

    return {
        "type_document": type_doc,
        "confiance"    : probas.max(),
        "top_3"        : top_3,
        "mots_cles"    : mots_cles,
        "probas"       : dict(zip(le.classes_, probas))
    }

def resumer_texte(texte, tfidf, n_phrases=4):
    import re
    phrases = re.split(r"(?<=[.!?])\s+", texte.strip())
    phrases = [p.strip() for p in phrases if len(p.split()) >= 5]
    if len(phrases) <= n_phrases:
        return texte
    vec    = tfidf.transform(phrases)
    scores = np.asarray(vec.sum(axis=1)).flatten()
    top_idx = sorted(scores.argsort()[::-1][:n_phrases])
    return " ".join([phrases[i] for i in top_idx])

def calculer_score_risque(probas, type_doc, texte):
    import re

    PATTERNS = {
        "Limitation de responsabilite": [
            # Francais
            "non responsable", "exclut toute responsabilite",
            "en aucun cas", "sans garantie", "exclut toute garantie",
            "limite sa responsabilite", "sous reserve",
            "dans les limites permises par la loi",
            # Anglais
            "not liable", "no liability", "without warranty",
            "as is", "disclaim all warranties", "in no event",
            "shall not be responsible", "without any guarantee",
            "to the fullest extent permitted", "limit our liability",
            "not responsible for", "no warranties",
        ],
        "Cession de donnees": [
            # Francais
            "cede a des tiers", "partage avec des partenaires",
            "a des fins publicitaires", "partenaires commerciaux",
            # Anglais
            "share with third parties", "third-party partners",
            "advertising purposes", "share your data",
            "disclose to third parties", "sell your data",
            "transfer your information", "third party service providers",
            "share your personal information", "marketing partners",
            "data sharing", "share with our partners",
        ],
        "Resiliation abusive": [
            # Francais
            "sans preavis", "resiliation immediate",
            "sans indemnite", "a sa seule discretion",
            "sans notification", "sans remboursement",
            # Anglais
            "without notice", "immediate termination",
            "at our sole discretion", "without refund",
            "terminate your account", "suspend or terminate",
            "without liability", "at any time without",
            "reserve the right to terminate", "without prior notice",
        ],
        "Engagement excessif": [
            # Francais
            "irrevocablement", "perpetuellement",
            "licence mondiale", "sans limitation de duree",
            # Anglais
            "irrevocably", "perpetually", "worldwide license",
            "royalty-free", "sublicensable", "irrevocable license",
            "unconditional right", "unlimited license",
            "worldwide, royalty-free", "in perpetuity",
            "non-exclusive, worldwide", "waive your right",
        ],
    }

    import unicodedata

    def normaliser(s):
        """Supprime accents et met en minuscules pour comparaison robuste."""
        return unicodedata.normalize("NFD", s.lower()).encode("ascii", "ignore").decode()

    texte_norm = normaliser(texte)
    clauses_trouvees = {}

    for categorie, mots in PATTERNS.items():
        extraits = []
        for mot in mots:
            mot_norm = normaliser(mot)
            idx = texte_norm.find(mot_norm)
            if idx != -1:
                debut   = max(0, idx - 50)
                fin     = min(len(texte), idx + len(mot) + 80)
                extrait = "..." + texte[debut:fin].strip() + "..."
                extraits.append((mot, extrait))
        if extraits:
            clauses_trouvees[categorie] = extraits

    nb_categories  = len(clauses_trouvees)
    nb_total       = sum(len(v) for v in clauses_trouvees.values())
    score_clauses  = min(nb_categories * 0.2 + nb_total * 0.05, 0.9)
    confiance      = max(probas.values())
    penalite       = (1 - confiance) * 0.1
    score          = min(score_clauses + penalite, 1.0)

    if score >= 0.55:
        niveau, couleur = "Risque eleve",  "#EF4444"
    elif score >= 0.25:
        niveau, couleur = "Risque modere", "#F59E0B"
    else:
        niveau, couleur = "Risque faible", "#10B981"

    return round(score, 2), niveau, couleur, clauses_trouvees

# INTERFACE PRINCIPALE

with st.sidebar:
    st.markdown("## Analyse Juridique IA")
    st.markdown("---")
    page = st.radio("Navigation", ["Analyser un document",
                                   "Tableau de bord",
                                   "Mots-clés par classe"])
    st.markdown("---")
    st.markdown("**Modèle :** CamemBERT-large")
    st.markdown("**Accuracy :** 94.49%")
    st.markdown("**Classes :** 14 types de documents")
    st.markdown("**Dataset :** data.gouv.fr")

# Chargement
try:
    clf, le, tfidf, model = charger_modeles()
    df_keywords           = charger_keywords()
    modeles_ok = True
except Exception as e:
    st.error(f"Erreur de chargement : {e}")
    modeles_ok = False

# PAGE 1 — ANALYSER UN DOCUMENT

if page == "Analyser un document":

    st.markdown('<p class="main-title">Analyse automatique de documents juridiques</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Identifiez le type de document et extrayez les mots-clés juridiques en un instant.</p>', unsafe_allow_html=True)

    mode = st.radio("Mode de saisie", ["Coller du texte", "Uploader un fichier .txt"],
                    horizontal=True)

    texte_input = ""

    if mode == "Coller du texte":
        texte_input = st.text_area(
            "Collez votre texte juridique ici",
            height=250,
            placeholder="Ex : Les présentes conditions générales de vente régissent les relations contractuelles..."
        )
    else:
        fichier = st.file_uploader("Choisir un fichier .txt", type=["txt"])
        if fichier:
            texte_input = fichier.read().decode("utf-8", errors="ignore")
            st.success(f"Fichier chargé : {fichier.name} ({len(texte_input):,} caractères)")
            with st.expander("Aperçu du texte"):
                st.text(texte_input[:800] + "..." if len(texte_input) > 800 else texte_input)

    col_btn, _ = st.columns([1, 4])
    with col_btn:
        analyser = st.button("Analyser", type="primary")

    if analyser and texte_input.strip() and modeles_ok:

        if len(texte_input.split()) < 10:
            st.markdown('<div class="warning-box">Texte trop court. Entrez au moins quelques phrases.</div>', unsafe_allow_html=True)
        else:
            with st.spinner("Analyse en cours..."):
                result = analyser_document(texte_input, clf, le, tfidf, model)

            st.markdown("---")
            st.markdown("### Résultats")

            # Métriques principales
            c1, c2, c3 = st.columns(3)
            with c1:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-val">{result['type_document']}</div>
                    <div class="metric-label">Type de document détecté</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-val">{result['confiance']:.1%}</div>
                    <div class="metric-label">Score de confiance</div>
                </div>""", unsafe_allow_html=True)
            with c3:
                st.markdown(f"""<div class="metric-box">
                    <div class="metric-val">{len(texte_input.split()):,}</div>
                    <div class="metric-label">Mots dans le document</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            col_left, col_right = st.columns([1, 1])

            # Top 3 probabilités
            with col_left:
                st.markdown("#### Top 3 classes probables")
                for classe, proba in result["top_3"]:
                    st.markdown(f"**{classe}**")
                    st.progress(float(proba), text=f"{proba:.1%}")

            # Mots-clés extraits
            with col_right:
                st.markdown("#### Mots-clés juridiques extraits")
                tags_html = "".join([f'<span class="tag">{m}</span>' for m in result["mots_cles"]])
                st.markdown(tags_html, unsafe_allow_html=True)

            # Résumé automatique
            st.markdown("---")
            st.markdown("#### Resume automatique")
            with st.spinner("Génération du résumé..."):
                resume = resumer_texte(texte_input, tfidf)
            st.info(resume)

            # Score de risque
            score_risque, niveau_risque, couleur_risque, clauses = calculer_score_risque(
                result["probas"], result["type_document"], texte_input
            )
            st.markdown("#### Score de risque du document")
            col_r1, col_r2, col_r3 = st.columns([1, 2, 2])
            with col_r1:
                st.markdown(f"""<div class="metric-box" style="border-left-color:{couleur_risque}">
                    <div class="metric-val" style="color:{couleur_risque}">{score_risque:.0%}</div>
                    <div class="metric-label">{niveau_risque}</div>
                </div>""", unsafe_allow_html=True)
            with col_r2:
                fig_r, ax_r = plt.subplots(figsize=(5, 0.6))
                ax_r.barh(0, 1,            color="#E5E7EB", height=0.5)
                ax_r.barh(0, score_risque, color=couleur_risque, height=0.5)
                ax_r.set_xlim(0, 1); ax_r.axis("off")
                fig_r.patch.set_alpha(0)
                st.pyplot(fig_r)
                plt.close()
            with col_r3:
                st.markdown(f"""
                **Type :** {result["type_document"]}  
                **Confiance :** {result["confiance"]:.1%}  
                **Risque :** {niveau_risque}  
                **Clauses detectees :** {sum(len(v) for v in clauses.values())}
                """)

            # Clauses risquees detectees
            if clauses:
                st.markdown("#### Clauses à risque detectées dans le texte")
                for categorie, extraits in clauses.items():
                    couleur_cat = {
                        "Limitation de responsabilite": "#EF4444",
                        "Cession de donnees"          : "#F59E0B",
                        "Resiliation abusive"         : "#EF4444",
                        "Engagement excessif"         : "#F59E0B",
                    }.get(categorie, "#6B7280")
                    st.markdown(
                        f"""<div style="border-left:4px solid {couleur_cat};
                        padding:0.6rem 1rem; margin:0.4rem 0;
                        background:#ff4d4d; border-radius:4px">
                        <strong>{categorie}</strong> — {len(extraits)} occurrence(s)</div>""",
                        unsafe_allow_html=True
                    )
                    for mot, extrait in extraits[:2]:
                        st.markdown(
                            f"""<div style="margin-left:1.5rem; font-size:0.85rem;
                            color:#6B7280; font-style:italic;
                            border-left:2px solid #E5E7EB; padding-left:0.8rem">
                            Mot detecte : <strong>{mot}</strong><br>{extrait}</div>""",
                            unsafe_allow_html=True
                        )
            else:
                st.markdown(
                    """<div style="background:#52f2a0; border-left:4px solid #10B981;
                    padding:0.8rem; border-radius:6px">
                    Aucune clause a risque detectee dans ce document.</div>""",
                    unsafe_allow_html=True
                )

            # Distribution complète des probabilités
            st.markdown("#### Distribution complète des probabilités")
            probas_df = pd.DataFrame(result["probas"].items(), columns=["Classe", "Probabilité"])
            probas_df = probas_df.sort_values("Probabilité", ascending=True)

            fig, ax = plt.subplots(figsize=(10, 5))
            colors = ["#2563EB" if c == result["type_document"] else "#CBD5E1"
                      for c in probas_df["Classe"]]
            ax.barh(probas_df["Classe"], probas_df["Probabilité"],
                    color=colors, edgecolor="white")
            ax.set_xlim(0, 1)
            ax.set_xlabel("Probabilité")
            ax.set_title("Probabilités par classe")
            for i, (_, row) in enumerate(probas_df.iterrows()):
                if row["Probabilité"] > 0.01:
                    ax.text(row["Probabilité"] + 0.01, i,
                            f"{row['Probabilité']:.1%}", va="center", fontsize=8)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

# PAGE 2 — TABLEAU DE BORD

elif page == "Tableau de bord":

    st.markdown('<p class="main-title">Tableau de bord du modèle</p>', unsafe_allow_html=True)

    # Métriques globales
    st.markdown("### Performance globale")
    c1, c2, c3, c4 = st.columns(4)
    metrics = [("94.49%", "Accuracy"), ("96.91%", "Macro F1"),
               ("99.90%", "Confiance médiane"), ("14", "Classes")]
    for col, (val, label) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""<div class="metric-box">
                <div class="metric-val">{val}</div>
                <div class="metric-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Graphiques d'évaluation")

    graphiques = {
        "Matrice de confusion normalisée" : "eval_confusion_normalisee.png",
        "F1-score par classe"             : "eval_f1_par_classe.png",
        "Courbes ROC"                     : "eval_roc_curves.png",
        "Distribution de confiance"       : "eval_confiance.png",
    }

    for titre, fichier in graphiques.items():
        chemin = os.path.join(OUTPUT_DIR, fichier)
        if os.path.exists(chemin):
            with st.expander(f"{titre}", expanded=False):
                img = Image.open(chemin)
                st.image(img, width=900)
        else:
            st.warning(f"Graphique non trouvé : {fichier}")

# PAGE 3 — MOTS-CLÉS PAR CLASSE

elif page == "Mots-clés par classe":

    st.markdown('<p class="main-title">Mots-clés juridiques par type de document</p>', unsafe_allow_html=True)

    if modeles_ok:
        classe_choisie = st.selectbox("Choisir un type de document",
                                       sorted(df_keywords["type_document"].unique()))

        df_classe = df_keywords[df_keywords["type_document"] == classe_choisie].head(15)

        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown(f"#### Top mots-clés — {classe_choisie}")
            tags_html = "".join([f'<span class="tag">{row.mot_cle}</span>'
                                  for _, row in df_classe.iterrows()])
            st.markdown(tags_html, unsafe_allow_html=True)

        with col2:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.barh(df_classe["mot_cle"][::-1], df_classe["score_tfidf"][::-1],
                    color="#2563EB", edgecolor="white", alpha=0.85)
            ax.set_title(f"Score TF-IDF — {classe_choisie}", fontsize=12)
            ax.set_xlabel("Score TF-IDF moyen")
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        st.markdown("---")
        st.markdown("#### Tableau complet")
        st.dataframe(df_classe[["rang","mot_cle","score_tfidf"]].reset_index(drop=True),
                     width=900, height=400)