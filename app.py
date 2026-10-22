import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import difflib

# ==========================================
# CONFIGURATION DE LA PAGE
# ==========================================
st.set_page_config(
    page_title="Foot Predictor Pro - Universel & Blessures",
    page_icon="⚽",
    layout="wide"
)

# Configuration API football-data.org
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

MOY_EUROPE_DOM = 1.65
MOY_EUROPE_EXT = 1.35

# ==========================================
# 1. DICTIONNAIRE ET CORRESPONDANCES DE NOMS
# ==========================================
CORRESPONDANCES_NOMS = {
    "Paris Saint-Germain FC": "Paris SG",
    "Paris Saint-Germain": "Paris SG",
    "Paris SG": "Paris SG",
    "FC Barcelona": "Barcelona",
    "Barcelona": "Barcelona",
    "Real Madrid CF": "Real Madrid",
    "Club Atlético de Madrid": "Ath Madrid",
    "Atletico Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao",
    "Athletic Bilbao": "Ath Bilbao",
    "FC Internazionale Milano": "Inter",
    "Inter Milan": "Inter",
    "AC Milan": "Milan",
    "Juventus FC": "Juventus",
    "AS Roma": "Roma",
    "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli",
    "Atalanta BC": "Atalanta",
    "AC Monza": "Monza",
    "Torino FC": "Torino",
    "FC Bayern München": "Bayern Munich",
    "Bayern Munich": "Bayern Munich",
    "Borussia Dortmund": "Dortmund",
    "Bayer 04 Leverkusen": "Leverkusen",
    "RB Leipzig": "RB Leipzig",
    "VfB Stuttgart": "Stuttgart",
    "Eintracht Frankfurt": "Ein Frankfurt",
    "Arsenal FC": "Arsenal",
    "Chelsea FC": "Chelsea",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Man City",
    "Manchester United FC": "Man United",
    "Tottenham Hotspur FC": "Tottenham",
    "Newcastle United FC": "Newcastle",
    "Aston Villa FC": "Aston Villa",
    "Olympique de Marseille": "Marseille",
    "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco",
    "LOSC Lille": "Lille",
    "OGC Nice": "Nice",
    "Stade Rennais FC 1904": "Rennes",
    "RC Lens": "Lens"
}

def trouver_nom_csv(nom_api, liste_equipes_csv):
    """Associe de façon fluide le nom de l'API au nom présent dans le fichier CSV."""
    if not nom_api:
        return liste_equipes_csv[0]
        
    if nom_api in CORRESPONDANCES_NOMS and CORRESPONDANCES_NOMS[nom_api] in liste_equipes_csv:
        return CORRESPONDANCES_NOMS[nom_api]

    nom_nettoye = nom_api.replace("FC", "").replace("CF", "").replace("AS", "").replace("SSC", "").strip()
    
    for eq in liste_equipes_csv:
        if eq.lower() == nom_nettoye.lower() or eq.lower() == nom_api.lower():
            return eq

    for eq in liste_equipes_csv:
        if eq.lower() in nom_nettoye.lower() or nom_nettoye.lower() in eq.lower():
            return eq

    proches = difflib.get_close_matches(nom_nettoye, liste_equipes_csv, n=1, cutoff=0.30)
    if proches:
        return proches[0]

    return liste_equipes_csv[0]

# ==========================================
# 2. CHARGEMENT DE L'HISTORIQUE CSV
# ==========================================
@st.cache_data
def charger_historique():
    df = pd.read_csv("historique_europe_complet.csv")
    return df

df_total = charger_historique()

def obtenir_forces_equipe(equipe):
    """Calcule les coefficients de force relative d'une équipe."""
    matchs = df_total[(df_total['Domicile'] == equipe) | (df_total['Exterieur'] == equipe)]
    if len(matchs) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.5, 1.2
    
    nom_ligue = matchs['Championnat'].iloc[0]
    df_ligue = df_total[df_total['Championnat'] == nom_ligue]
    
    moy_dom = df_ligue['Buts_Dom'].mean()
    moy_ext = df_ligue['Buts_Ext'].mean()

    m_dom = df_ligue[df_ligue['Domicile'] == equipe]
    m_ext = df_ligue[df_ligue['Exterieur'] == equipe]

    att_d = (m_dom['Buts_Dom'].mean() / moy_dom) if len(m_dom) > 0 else 1.0
    def_d = (m_dom['Buts_Ext'].mean() / moy_ext) if len(m_dom) > 0 else 1.0
    att_e = (m_ext['Buts_Ext'].mean() / moy_ext) if len(m_ext) > 0 else 1.0
    def_e = (m_ext['Buts_Dom'].mean() / moy_dom) if len(m_ext) > 0 else 1.0

    return att_d, def_d, att_e, def_e, moy_dom, moy_ext

# ==========================================
# 3. MODÈLE MATHÉMATIQUE (POISSON / MAHER)
# ==========================================
def predire_rencontre(equipe_dom, equipe_ext, est_europe=False, malus_dom=0.0, malus_ext=0.0):
    att_d, def_d, _, _, moy_dom_nat, _ = obtenir_forces_equipe(equipe_dom)
    _, _, att_e, def_e, _, moy_ext_nat = obtenir_forces_equipe(equipe_ext)

    base_dom = MOY_EUROPE_DOM if est_europe else moy_dom_nat
    base_ext = MOY_EUROPE_EXT if est_europe else moy_ext_nat

    # Application de l'impact des blessures / actualité
    lambda_dom = max(0.20, (att_d * def_e * base_dom) - malus_dom)
    lambda_ext = max(0.20, (att_e * def_d * base_ext) - malus_ext)

    mat_ft = np.zeros((7, 7))
    scores_dict = {}
    for i in range(7):
        for j in range(7):
            p = poisson.pmf(i, lambda_dom) * poisson.pmf(j, lambda_ext)
            mat_ft[i, j] = p
            scores_dict[f"{i} - {j}"] = p

    p_v1 = np.sum(np.tril(mat_ft, -1)) * 100
    p_x  = np.sum(np.diag(mat_ft)) * 100
    p_v2 = np.sum(np.triu(mat_ft, 1)) * 100

    over_15 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 1.5) * 100
    over_25 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 2.5) * 100
    over_35 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 3.5) * 100
    btts_oui = np.sum(mat_ft[1:, 1:]) * 100

    top_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    # Marchés 1ère Mi-Temps (44% des buts en moyenne)
    l_d_mt = 0.44 * lambda_dom
    l_e_mt = 0.44 * lambda_ext
    mat_mt = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            mat_mt[i, j] = poisson.pmf(i, l_d_mt) * poisson.pmf(j, l_e_mt)

    over_05_mt = (1.0 - mat_mt[0, 0]) * 100
    v1_mt = np.sum(np.tril(mat_mt, -1)) * 100
    x_mt  = np.sum(np.diag(mat_mt)) * 100
    v2_mt = np.sum(np.triu(mat_mt, 1)) * 100

    # Cotes équitables (Fair Odds)
    cote_v1 = (100.0 / p_v1) if p_v1 > 0 else 99.0
    cote_x  = (100.0 / p_x)  if p_x > 0 else 99.0
    cote_v2 = (100.0 / p_v2) if p_v2 > 0 else 99.0
    cote_over25 = (100.0 / over_25) if over_25 > 0 else 99.0
    cote_btts = (100.0 / btts_oui) if btts_oui > 0 else 99.0

    return {
        "lambda_dom": lambda_dom,
        "lambda_ext": lambda_ext,
        "1N2": (p_v1, p_x, p_v2),
        "cotes_1N2": (cote_v1, cote_x, cote_v2),
        "over_15": over_15, "over_25": over_25, "over_35": over_35,
        "cote_over25": cote_over25,
        "btts": btts_oui,
        "cote_btts": cote_btts,
        "top_scores": top_scores,
        "over_05_mt": over_05_mt,
        "1N2_mt": (v1_mt, x_mt, v2_mt)
    }

# ==========================================
# 4. APPEL CALENDRIER API & SCANNER
# ==========================================
@st.cache_data(ttl=1800)
def recuperer_matchs_api(code_compet):
    url = f"https://api.football-data.org/v4/competitions/{code_compet}/matches?status=SCHEDULED"
    try:
        req = requests.get(url, headers=HEADERS, timeout=6)
        data = req.json()
        matchs = []
        for m in data.get("matches", [])[:15]:
            matchs.append({
                "domicile_api": m["homeTeam"]["name"],
                "exterieur_api": m["awayTeam"]["name"],
                "date": m["utcDate"][:16].replace("T", " à ")
            })
        return matchs
    except Exception:
        return []

def scanner_toutes_competitions():
    opportunites = []
    toutes_eq = sorted(list(set(df_total['Domicile'].unique()).union(set(df_total['Exterieur'].unique()))))

    for nom_compet, code in CODES_COMPETITIONS.items():
        matchs = recuperer_matchs_api(code)
        est_europe = (code == "CL")
        
        for m in matchs:
            dom = trouver_nom_csv(m['domicile_api'], toutes_eq)
            ext = trouver_nom_csv(m['exterieur_api'], toutes_eq)
            res = predire_rencontre(dom, ext, est_europe=est_europe)

            if res['1N2'][0] >= 65.0:
                opportunites.append({
                    "Match": f"{dom} vs {ext}",
                    "Date": m['date'],
                    "Ligue": nom_compet,
                    "Pari Détecté": f"Victoire {dom} (1)",
                    "Probabilité": f"{res['1N2'][0]:.1f} %",
                    "Cote Min": f"@{res['cotes_1N2'][0]:.2f}",
                    "xG": f"{res['lambda_dom']:.2f} - {res['lambda_ext']:.2f}"
                })
            elif res['1N2'][2] >= 60.0:
                opportunites.append({
                    "Match": f"{dom} vs {ext}",
                    "Date": m['date'],
                    "Ligue": nom_compet,
                    "Pari Détecté": f"Victoire {ext} (2)",
                    "Probabilité": f"{res['1N2'][2]:.1f} %",
                    "Cote Min": f"@{res['cotes_1N2'][2]:.2f}",
                    "xG": f"{res['lambda_dom']:.2f} - {res['lambda_ext']:.2f}"
                })

            if res['over_25'] >= 64.0:
                opportunites.append({
                    "Match": f"{dom} vs {ext}",
                    "Date": m['date'],
                    "Ligue": nom_compet,
                    "Pari Détecté": "Over 2.5 Buts",
                    "Probabilité": f"{res['over_25']:.1f} %",
                    "Cote Min": f"@{res['cote_over25']:.2f}",
                    "xG": f"{res['lambda_dom']:.2f} - {res['lambda_ext']:.2f}"
                })

            if res['btts'] >= 60.0:
                opportunites.append({
                    "Match": f"{dom} vs {ext}",
                    "Date": m['date'],
                    "Ligue": nom_compet,
                    "Pari Détecté": "BTTS (Oui)",
                    "Probabilité": f"{res['btts']:.1f} %",
                    "Cote Min": f"@{res['cote_btts']:.2f}",
                    "xG": f"{res['lambda_dom']:.2f} - {res['lambda_ext']:.2f}"
                })

    return pd.DataFrame(opportunites)

# ==========================================
# 5. NAVIGATION PAR ONGLETS
# ==========================================
st.title("⚽ Foot Predictor Pro")
st.caption("Modélisation probabiliste (Poisson & Maher), Impact Blessures & Scanner de la semaine.")

onglet_match, onglet_top_picks = st.tabs(["🔍 Analyse par Match", "⭐ Top Sélections de la Semaine"])

toutes_equipes = sorted(list(set(df_total['Domicile'].unique()).union(set(df_total['Exterieur'].unique()))))

# ==========================================
# ONGLET 1 : ANALYSE PAR MATCH
# ==========================================
with onglet_match:
    st.sidebar.header("🏆 Compétition & Match")
    categorie = st.sidebar.radio("Catégorie :", ["Championnats Nationaux", "Coupes d'Europe (UCL, UEL, UECL)"])

    dom_selectionne, ext_selectionne = None, None
    est_europeen = False
    date_info = ""

    if categorie == "Championnats Nationaux":
        ligues_dispos = [k for k in CODES_COMPETITIONS.keys() if k != "UEFA Champions League"]
        ligue_choisie = st.sidebar.selectbox("Championnat :", ligues_dispos)
        code_api = CODES_COMPETITIONS[ligue_choisie]
        df_ligue = df_total[df_total['Championnat'] == ligue_choisie]
        equipes_ligue = sorted(list(set(df_ligue['Domicile'].unique()).union(set(df_ligue['Exterieur'].unique()))))

        mode = st.sidebar.radio("Mode d'analyse :", ["Matchs à venir (Calendrier officiel)", "Simulation manuelle libre"])

        if mode == "Matchs à venir (Calendrier officiel)":
            matchs = recuperer_matchs_api(code_api)
            if not matchs:
                st.warning("Aucune rencontre à venir n'est programmée pour cette ligue.")
            else:
                options_matchs = {f"{m['domicile_api']} vs {m['exterieur_api']} — ({m['date']})": m for m in matchs}
                choix = st.selectbox("Sélectionnez l'affiche :", list(options_matchs.keys()))
                m_choisi = options_matchs[choix]
                
                dom_selectionne = trouver_nom_csv(m_choisi['domicile_api'], equipes_ligue)
                ext_selectionne = trouver_nom_csv(m_choisi['exterieur_api'], equipes_ligue)
                date_info = m_choisi['date']
        else:
            st.subheader("🧪 Sélection manuelle des équipes")
            c1, c2 = st.columns(2)
            dom_selectionne = c1.selectbox("Équipe Domicile :", equipes_ligue, index=0)
            ext_selectionne = c2.selectbox("Équipe Extérieur :", [e for e in equipes_ligue if e != dom_selectionne], index=0)

    else:
        tournoi = st.sidebar.selectbox("Tournoi Européen :", [
            "UEFA Champions League", 
            "UEFA Europa League", 
            "UEFA Europa Conference League"
        ])
        est_europeen = True

        if tournoi == "UEFA Champions League":
            mode_ucl = st.sidebar.radio("Mode UCL :", ["Matchs à venir (Calendrier UCL)", "Simulation manuelle libre"])
            if mode_ucl == "Matchs à venir (Calendrier UCL)":
                matchs_ucl = recuperer_matchs_api("CL")
                if not matchs_ucl:
                    st.info("Aucun match de Champions League programmé dans les prochains jours.")
                else:
                    opts_ucl = {f"{m['domicile_api']} vs {m['exterieur_api']} — ({m['date']})": m for m in matchs_ucl}
                    ch = st.selectbox("Sélectionnez l'affiche de Champions League :", list(opts_ucl.keys()))
                    m_sel = opts_ucl[ch]
                    
                    dom_selectionne = trouver_nom_csv(m_sel['domicile_api'], toutes_equipes)
                    ext_selectionne = trouver_nom_csv(m_sel['exterieur_api'], toutes_equipes)
                    date_info = m_sel['date']

        if not dom_selectionne:
            st.subheader(f"⚔️ Simulation d'affiche : {tournoi}")
            c1, c2 = st.columns(2)
            dom_selectionne = c1.selectbox("Équipe Domicile (Europe) :", toutes_equipes, index=0)
            ext_selectionne = c2.selectbox("Équipe Extérieur (Europe) :", [e for e in toutes_equipes if e != dom_selectionne], index=1)

    # Paramètres d'actualité et de blessures dans la barre latérale
    if dom_selectionne and ext_selectionne:
        st.sidebar.markdown("---")
        st.sidebar.header("🏥 Actualité & Absences")
        st.sidebar.caption("Ajustez si un buteur majeur est blessé ou si l'équipe a joué en milieu de semaine.")
        
        malus_dom = st.sidebar.slider(
            f"Impact absences / fatigue pour {dom_selectionne} :", 
            0.0, 0.50, 0.0, 0.05,
            help="Réduit l'espérance de buts attendue de l'équipe recevant."
        )
        malus_ext = st.sidebar.slider(
            f"Impact absences / fatigue pour {ext_selectionne} :", 
            0.0, 0.50, 0.0, 0.05,
            help="Réduit l'espérance de buts attendue de l'équipe en déplacement."
        )

        res = predire_rencontre(dom_selectionne, ext_selectionne, est_europe=est_europeen, malus_dom=malus_dom, malus_ext=malus_ext)

        st.markdown("---")
        st.subheader(f"🏟️ {dom_selectionne}   vs   {ext_selectionne}")
        if date_info:
            st.caption(f"📅 **Date officielle :** {date_info}")

        if malus_dom > 0 or malus_ext > 0:
            st.warning(f"⚠️ **Ajustement d'actualité actif :** Malus {dom_selectionne} (-{malus_dom:.2f} xG) | Malus {ext_selectionne} (-{malus_ext:.2f} xG)")

        # Espérances de buts xG
        c_xg1, c_xg2 = st.columns(2)
        c_xg1.metric(f"xG Estimé ({dom_selectionne})", f"{res['lambda_dom']:.2f} buts")
        c_xg2.metric(f"xG Estimé ({ext_selectionne})", f"{res['lambda_ext']:.2f} buts")

        # 1N2 avec cotes théoriques
        st.markdown("### 🎯 Issue Finale (1N2) & Cotes Théoriques")
        c_v1, c_x, c_v2 = st.columns(3)
        c_v1.success(f"**Victoire {dom_selectionne} (1)**\n\n### {res['1N2'][0]:.1f} %\n*Cote min : @{res['cotes_1N2'][0]:.2f}*")
        c_x.info(f"**Match Nul (X)**\n\n### {res['1N2'][1]:.1f} %\n*Cote min : @{res['cotes_1N2'][1]:.2f}*")
        c_v2.error(f"**Victoire {ext_selectionne} (2)**\n\n### {res['1N2'][2]:.1f} %\n*Cote min : @{res['cotes_1N2'][2]:.2f}*")

        st.markdown("---")
        col_gauche, col_droite = st.columns(2)

        with col_gauche:
            st.markdown("### 🤝 Les Deux Équipes Marquent (BTTS)")
            p_oui = res['btts']
            verdict_btts = "OUI" if p_oui >= 50 else "NON"
            st.info(f"Tendance statistique : **{verdict_btts}** (Cote min OUI : @{res['cote_btts']:.2f})")
            
            cb1, cb2 = st.columns(2)
            cb1.metric("OUI", f"{p_oui:.1f} %")
            cb2.metric("NON", f"{100.0 - p_oui:.1f} %")

            st.markdown("### 📈 Marchés de Buts (Over / Under)")
            st.write(f"• **Over 1.5** : `{res['over_15']:.1f}%`  |  **Under 1.5** : `{100 - res['over_15']:.1f}%`")
            st.write(f"• **Over 2.5** : `{res['over_25']:.1f}%` *(Cote @{res['cote_over25']:.2f})*  |  **Under 2.5** : `{100 - res['over_25']:.1f}%`")
            st.write(f"• **Over 3.5** : `{res['over_35']:.1f}%`  |  **Under 3.5** : `{100 - res['over_35']:.1f}%`")

        with col_droite:
            st.markdown("### ⏱️ Marchés 1ère Mi-Temps")
            st.write(f"• **Au moins 1 but en 1ère MT (+0.5)** : `{res['over_05_mt']:.1f} %`")
            st.write(f"• **1N2 à la pause** : 1 (`{res['1N2_mt'][0]:.1f}%`) | Nul (`{res['1N2_mt'][1]:.1f}%`) | 2 (`{res['1N2_mt'][2]:.1f}%`)")

            st.markdown("### 🎲 Top 5 Scores Exacts les Plus Probables")
            df_sc = pd.DataFrame([{"Score": s, "Probabilité": f"{p * 100:.1f} %"} for s, p in res["top_scores"]])
            st.table(df_sc)

# ==========================================
# ONGLET 2 : SCANNER DES MEILLEURES SÉLECTIONS
# ==========================================
with onglet_top_picks:
    st.subheader("🔥 Meilleures Sélections Statistiques de la Semaine")
    st.markdown("Ce scanner analyse les matchs programmés sur les 5 grands championnats et l'UEFA Champions League pour repérer les probabilités élevées.")

    if st.button("🚀 Scanner toutes les compétitions maintenant", use_container_width=True):
        with st.spinner("Analyse statistique des calendriers en cours..."):
            df_opportunites = scanner_toutes_competitions()
            
            if not df_opportunites.empty:
                st.success(f"Analyse terminée : {len(df_opportunites)} opportunités majeures identifiées !")
                st.dataframe(df_opportunites, use_container_width=True)
            else:
                st.warning("Aucun match de la semaine ne dépasse actuellement les seuils stricts de haute confiance.")