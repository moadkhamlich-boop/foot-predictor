import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION ET THEME SOMBRE
# ==========================================
st.set_page_config(page_title="Foot Predictor V2", page_icon="⚽", layout="centered")

API_TOKEN = "154e2f8f245f4d07b64977f1bf2d9ec9"
HEADERS = {"X-Auth-Token": API_TOKEN}

CODES_COMPETITIONS = {
    "UEFA Champions League": "CL",
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1"
}

# ==========================================
# 1. OUTILS ET CORRESPONDANCES
# ==========================================
def trouver_nom_csv(nom_api, liste_equipes_csv):
    nom_nettoye = nom_api.replace("FC", "").replace("CF", "").replace("AS", "").replace("SSC", "").strip()
    for eq in liste_equipes_csv:
        if eq.lower() == nom_nettoye.lower() or eq.lower() == nom_api.lower() or eq.lower() in nom_nettoye.lower():
            return eq
    proches = difflib.get_close_matches(nom_nettoye, liste_equipes_csv, n=1, cutoff=0.35)
    return proches[0] if proches else liste_equipes_csv[0]

# ==========================================
# 2. CHARGEMENT ET CALCULS AVANCÉS (FORME RÉCENTE)
# ==========================================
@st.cache_data
def charger_historique():
    return pd.read_csv("historique_europe_complet.csv")

df_total = charger_historique()

def calculer_forces_avancees(equipe):
    """Calcule la force historique (60%) combinée à la forme des 6 derniers matchs (40%)."""
    matchs = df_total[(df_total['Domicile'] == equipe) | (df_total['Exterieur'] == equipe)]
    if len(matchs) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.5, 1.2
    
    nom_ligue = matchs['Championnat'].iloc[0]
    df_ligue = df_total[df_total['Championnat'] == nom_ligue]
    
    moy_dom, moy_ext = df_ligue['Buts_Dom'].mean(), df_ligue['Buts_Ext'].mean()

    # Historique Global
    m_dom = df_ligue[df_ligue['Domicile'] == equipe]
    m_ext = df_ligue[df_ligue['Exterieur'] == equipe]

    att_d_hist = (m_dom['Buts_Dom'].mean() / moy_dom) if len(m_dom) > 0 else 1.0
    def_d_hist = (m_dom['Buts_Ext'].mean() / moy_ext) if len(m_dom) > 0 else 1.0
    att_e_hist = (m_ext['Buts_Ext'].mean() / moy_ext) if len(m_ext) > 0 else 1.0
    def_e_hist = (m_ext['Buts_Dom'].mean() / moy_dom) if len(m_ext) > 0 else 1.0

    # Dynamique (6 derniers matchs)
    m_dom_recents = m_dom.tail(6)
    m_ext_recents = m_ext.tail(6)

    att_d_form = (m_dom_recents['Buts_Dom'].mean() / moy_dom) if len(m_dom_recents) > 0 else att_d_hist
    def_d_form = (m_dom_recents['Buts_Ext'].mean() / moy_ext) if len(m_dom_recents) > 0 else def_d_hist
    att_e_form = (m_ext_recents['Buts_Ext'].mean() / moy_ext) if len(m_ext_recents) > 0 else att_e_hist
    def_e_form = (m_ext_recents['Buts_Dom'].mean() / moy_dom) if len(m_ext_recents) > 0 else def_e_hist

    # Pondération 60% Historique / 40% Forme du moment
    att_d = (att_d_hist * 0.6) + (att_d_form * 0.4)
    def_d = (def_d_hist * 0.6) + (def_d_form * 0.4)
    att_e = (att_e_hist * 0.6) + (att_e_form * 0.4)
    def_e = (def_e_hist * 0.6) + (def_e_form * 0.4)

    return att_d, def_d, att_e, def_e, moy_dom, moy_ext

# ==========================================
# 3. MODÈLE MATHÉMATIQUE 
# ==========================================
def predire_rencontre(equipe_dom, equipe_ext, est_europe=False):
    att_d, def_d, _, _, moy_dom_nat, _ = calculer_forces_avancees(equipe_dom)
    _, _, att_e, def_e, _, moy_ext_nat = calculer_forces_avancees(equipe_ext)

    base_dom = 1.65 if est_europe else moy_dom_nat
    base_ext = 1.35 if est_europe else moy_ext_nat

    lambda_dom = max(0.2, att_d * def_e * base_dom)
    lambda_ext = max(0.2, att_e * def_d * base_ext)

    mat_ft = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            mat_ft[i, j] = poisson.pmf(i, lambda_dom) * poisson.pmf(j, lambda_ext)

    p_v1 = np.sum(np.tril(mat_ft, -1)) * 100
    p_x  = np.sum(np.diag(mat_ft)) * 100
    p_v2 = np.sum(np.triu(mat_ft, 1)) * 100

    over_15 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 1.5) * 100
    over_25 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 2.5) * 100
    btts_oui = np.sum(mat_ft[1:, 1:]) * 100

    return {
        "xG_dom": lambda_dom, "xG_ext": lambda_ext,
        "1N2": (p_v1, p_x, p_v2),
        "over_15": over_15, "over_25": over_25, "btts": btts_oui
    }

# ==========================================
# 4. RÉCUPÉRATION API & FILTRE TEMPOREL
# ==========================================
@st.cache_data(ttl=1800)
def recuperer_matchs_par_ligue(code_compet):
    url = f"https://api.football-data.org/v4/competitions/{code_compet}/matches?status=SCHEDULED"
    try:
        req = requests.get(url, headers=HEADERS, timeout=6)
        return req.json().get("matches", [])[:10]
    except:
        return []

# ==========================================
# 5. INTERFACE MOBILE "FLASHSCORE"
# ==========================================
st.markdown("<h1 style='text-align: center;'>⚽ DIRECT FOOT</h1>", unsafe_allow_html=True)
st.caption("Filtre mathématique prédictif - Algorithme Forme Récente Actif")

# Barre de navigation temporelle
onglets_jours = st.tabs(["Aujourd'hui", "Demain", "Prochains Jours", "Top Value Bets"])
jours_filtre = [datetime.utcnow().date(), datetime.utcnow().date() + timedelta(days=1), None]

toutes_eq = sorted(list(set(df_total['Domicile'].unique()).union(set(df_total['Exterieur'].unique()))))

for idx_onglet, onglet in enumerate(onglets_jours[:3]):
    with onglet:
        date_cible = jours_filtre[idx_onglet]
        matchs_trouves = False

        for ligue_nom, code_api in CODES_COMPETITIONS.items():
            matchs_ligue = recuperer_matchs_par_ligue(code_api)
            
            # Filtrer les matchs selon l'onglet choisi
            if date_cible:
                matchs_filtres = [m for m in matchs_ligue if m["utcDate"].startswith(str(date_cible))]
            else:
                matchs_filtres = [m for m in matchs_ligue if not m["utcDate"].startswith(str(jours_filtre[0])) and not m["utcDate"].startswith(str(jours_filtre[1]))]

            if matchs_filtres:
                matchs_trouves = True
                # Affichage par Expanders (Blocs par compétition comme sur l'iPhone)
                with st.expander(f"🏆 {ligue_nom}", expanded=True):
                    for m in matchs_filtres:
                        dom_api = m["homeTeam"]["name"]
                        ext_api = m["awayTeam"]["name"]
                        heure = m["utcDate"][11:16]

                        # Bouton d'ouverture de l'analyse (Style carte match)
                        col1, col2 = st.columns([1, 4])
                        col1.write(f"**{heure}**")
                        if col2.button(f"{dom_api} - {ext_api}", key=m['id']):
                            st.markdown("---")
                            dom_csv = trouver_nom_csv(dom_api, toutes_eq)
                            ext_csv = trouver_nom_csv(ext_api, toutes_eq)
                            res = predire_rencontre(dom_csv, ext_csv, est_europe=(code_api == "CL"))
                            
                            st.success(f"**Analyse de : {dom_csv} vs {ext_csv}**")
                            c1, c2, c3 = st.columns(3)
                            c1.metric(f"V1 ({dom_csv})", f"{res['1N2'][0]:.1f}%")
                            c2.metric("Nul (X)", f"{res['1N2'][1]:.1f}%")
                            c3.metric(f"V2 ({ext_csv})", f"{res['1N2'][2]:.1f}%")

                            c4, c5, c6 = st.columns(3)
                            c4.metric("Over 2.5", f"{res['over_25']:.1f}%")
                            c5.metric("BTTS (Oui)", f"{res['btts']:.1f}%")
                            c6.metric("Score xG", f"{res['xG_dom']:.1f} - {res['xG_ext']:.1f}")
                            st.markdown("---")

        if not matchs_trouves:
            st.info("Aucun match majeur programmé à cette date.")

# Le 4e Onglet est réservé au futur tipster IA et au scanner
with onglets_jours[3]:
    st.markdown("### 🤖 Tipster Virtuel IA & Value Bets")
    st.info("Interface prête. L'intégration de l'Intelligence Artificielle (Gemini/OpenAI) sera branchée ici pour générer les résumés d'experts.")
