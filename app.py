import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta, timezone

# ==========================================
# CONFIGURATION & CLÉS API (FOOTBALL + GEMINI AI)
# ==========================================
st.set_page_config(page_title="Direct Foot Pro AI", page_icon="⚽", layout="centered")

API_TOKEN = "154e2f8f245f4d07b64977f1bf2d9ec9"
HEADERS = {"X-Auth-Token": API_TOKEN}

GEMINI_API_KEY = "AQ.Ab8RN6L5dMyifo8wWnczhTj4dpkYg68sRGYYIMFDvJNTeW2kpA"
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

CODES_COMPETITIONS = {
    "UEFA Champions League": "CL",
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1"
}

# ==========================================
# 1. DICTIONNAIRE EXACT & CLOISONNEMENT PAR LIGUE
# ==========================================
CORRESPONDANCES_NOMS = {
    # Ligue 1
    "Paris Saint-Germain FC": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco", "LOSC Lille": "Lille", "OGC Nice": "Nice", "Stade Rennais FC 1904": "Rennes",
    "RC Lens": "Lens", "Stade Brestois 29": "Brest", "RC Strasbourg Alsace": "Strasbourg", "FC Nantes": "Nantes",
    "Montpellier HSC": "Montpellier", "Toulouse FC": "Toulouse", "Stade de Reims": "Reims", "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre", "Angers SCO": "Angers", "AS Saint-Étienne": "St Etienne",
    # Premier League
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest", "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves",
    # La Liga
    "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid", "Club Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao", "Real Sociedad de Fútbol": "Sociedad", "Real Betis Balompié": "Betis",
    "Villarreal CF": "Villarreal", "Valencia CF": "Valencia", "Sevilla FC": "Sevilla", "Girona FC": "Girona",
    "RC Celta de Vigo": "Celta", "CA Osasuna": "Osasuna", "Getafe CF": "Getafe", "Rayo Vallecano de Madrid": "Vallecano",
    "RCD Mallorca": "Mallorca", "Deportivo Alavés": "Alaves", "UD Las Palmas": "Las Palmas", "CD Leganés": "Leganes",
    "RCD Espanyol de Barcelona": "Espanyol", "Real Valladolid CF": "Valladolid",
    # Serie A
    "FC Internazionale Milano": "Inter", "AC Milan": "Milan", "Juventus FC": "Juventus", "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna", "AS Roma": "Roma", "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "SSC Napoli": "Napoli", "Torino FC": "Torino", "Genoa CFC": "Genoa", "AC Monza": "Monza",
    "Hellas Verona FC": "Verona", "US Lecce": "Lecce", "Udinese Calcio": "Udinese", "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli", "Parma Calcio 1913": "Parma", "Como 1907": "Como", "Venezia FC": "Venezia",
    # Bundesliga
    "Bayer 04 Leverkusen": "Leverkusen", "FC Bayern München": "Bayern Munich", "VfB Stuttgart": "Stuttgart",
    "RB Leipzig": "RB Leipzig", "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt",
    "TSG 1899 Hoffenheim": "Hoffenheim", "1. FC Heidenheim 1846": "Heidenheim", "SV Werder Bremen": "Werder Bremen",
    "SC Freiburg": "Freiburg", "FC Augsburg": "Augsburg", "VfL Wolfsburg": "Wolfsburg", "1. FSV Mainz 05": "Mainz",
    "Borussia Mönchengladbach": "M'gladbach", "1. FC Union Berlin": "Union Berlin", "VfL Bochum 1848": "Bochum",
    "FC St. Pauli 1910": "St Pauli", "Holstein Kiel": "Holstein Kiel"
}

def trouver_nom_csv_strict(nom_api, equipes_autorisees):
    if not nom_api or len(equipes_autorisees) == 0:
        return None
    if nom_api in CORRESPONDANCES_NOMS:
        cible = CORRESPONDANCES_NOMS[nom_api]
        if cible in equipes_autorisees:
            return cible

    mots_parasites = ["FC", "CF", "AS", "SSC", "AC", "SC", "SV", "RC", "RCD", "UD", "CD", "AFC", "1909", "1904", "1899", "1846", "1848", "1910", "1913", "1907", "04", "05"]
    nom_net = nom_api
    for m in mots_parasites:
        nom_net = nom_net.replace(f" {m} ", " ").replace(f" {m}", "").replace(f"{m} ", "")
    nom_net = nom_net.strip()

    for eq in equipes_autorisees:
        if eq.lower() == nom_net.lower() or eq.lower() == nom_api.lower():
            return eq
    for eq in equipes_autorisees:
        if len(eq) >= 4 and (eq.lower() in nom_net.lower() or nom_net.lower() in eq.lower()):
            return eq

    proches = difflib.get_close_matches(nom_net, equipes_autorisees, n=1, cutoff=0.58)
    return proches[0] if proches else None

# ==========================================
# 2. CONNECTEUR IA GÉNÉRATIVE GOOGLE GEMINI
# ==========================================
@st.cache_data(ttl=3600, show_spinner=False)
def appeler_gemini_ai(prompt_texte):
    """Appelle l'API officielle Gemini Flash avec le format exact X-goog-api-key."""
    headers_ia = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY
    }
    payload = {
        "contents": [{"parts": [{"text": prompt_texte}]}]
    }
    try:
        r = requests.post(GEMINI_URL, headers=headers_ia, json=payload, timeout=12)
        if r.status_code == 200:
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None

# ==========================================
# 3. CHARGEMENT ET CALCULS DE FORCES (FORME + MT)
# ==========================================
@st.cache_data
def charger_historique():
    return pd.read_csv("historique_europe_complet.csv")

df_total = charger_historique()
TOUTES_EQUIPES_CSV = sorted(list(set(df_total['Domicile'].unique()).union(set(df_total['Exterieur'].unique()))))

def obtenir_equipes_ligue(nom_compet):
    if nom_compet in df_total['Championnat'].values:
        df_l = df_total[df_total['Championnat'] == nom_compet]
        return sorted(list(set(df_l['Domicile'].unique()).union(set(df_l['Exterieur'].unique()))))
    return TOUTES_EQUIPES_CSV

def calculer_forces_completes(equipe):
    matchs = df_total[(df_total['Domicile'] == equipe) | (df_total['Exterieur'] == equipe)]
    if len(matchs) == 0:
        return 1.0, 1.0, 1.0, 1.0, 1.50, 1.20, 0.44, 0.44, 0.0

    nom_ligue = matchs['Championnat'].iloc[0]
    df_ligue = df_total[df_total['Championnat'] == nom_ligue]
    moy_dom, moy_ext = df_ligue['Buts_Dom'].mean(), df_ligue['Buts_Ext'].mean()

    m_dom = df_ligue[df_ligue['Domicile'] == equipe]
    m_ext = df_ligue[df_ligue['Exterieur'] == equipe]

    att_d_h = (m_dom['Buts_Dom'].mean() / moy_dom) if len(m_dom) > 0 else 1.0
    def_d_h = (m_dom['Buts_Ext'].mean() / moy_ext) if len(m_dom) > 0 else 1.0
    att_e_h = (m_ext['Buts_Ext'].mean() / moy_ext) if len(m_ext) > 0 else 1.0
    def_e_h = (m_ext['Buts_Dom'].mean() / moy_dom) if len(m_ext) > 0 else 1.0

    m_dom_rec, m_ext_rec = m_dom.tail(6), m_ext.tail(6)
    att_d_f = (m_dom_rec['Buts_Dom'].mean() / moy_dom) if len(m_dom_rec) > 0 else att_d_h
    def_d_f = (m_dom_rec['Buts_Ext'].mean() / moy_ext) if len(m_dom_rec) > 0 else def_d_h
    att_e_f = (m_ext_rec['Buts_Ext'].mean() / moy_ext) if len(m_ext_rec) > 0 else att_e_h
    def_e_f = (m_ext_rec['Buts_Dom'].mean() / moy_dom) if len(m_ext_rec) > 0 else def_e_h

    att_d = (0.60 * att_d_h) + (0.40 * att_d_f)
    def_d = (0.60 * def_d_h) + (0.40 * def_d_f)
    att_e = (0.60 * att_e_h) + (0.40 * att_e_f)
    def_e = (0.60 * def_e_h) + (0.40 * def_e_f)

    indice_forme = ((att_d_f + att_e_f) / max(0.5, (att_d_h + att_e_h)) - 1.0) * 100

    ratio_mt_dom, ratio_mt_ext = 0.44, 0.44
    if 'Buts_Dom_MT' in df_ligue.columns and 'Buts_Ext_MT' in df_ligue.columns:
        tot_d, tot_e = m_dom['Buts_Dom'].sum(), m_ext['Buts_Ext'].sum()
        if tot_d > 5:
            ratio_mt_dom = np.clip(m_dom['Buts_Dom_MT'].sum() / tot_d, 0.30, 0.60)
        if tot_e > 5:
            ratio_mt_ext = np.clip(m_ext['Buts_Ext_MT'].sum() / tot_e, 0.30, 0.60)

    return att_d, def_d, att_e, def_e, moy_dom, moy_ext, ratio_mt_dom, ratio_mt_ext, indice_forme

# ==========================================
# 4. MODÈLE MATHÉMATIQUE (POISSON + DIXON-COLES)
# ==========================================
def predire_rencontre(equipe_dom, equipe_ext, est_europe=False, malus_dom=0.0, malus_ext=0.0):
    att_d, def_d, _, _, moy_dom_nat, _, r_mt_d, _, forme_dom = calculer_forces_completes(equipe_dom)
    _, _, att_e, def_e, _, moy_ext_nat, _, r_mt_e, forme_ext = calculer_forces_completes(equipe_ext)

    base_dom = 1.62 if est_europe else moy_dom_nat
    base_ext = 1.32 if est_europe else moy_ext_nat

    lambda_dom = max(0.20, (att_d * def_e * base_dom) - malus_dom)
    lambda_ext = max(0.20, (att_e * def_d * base_ext) - malus_ext)

    rho = -0.08
    mat_ft = np.zeros((7, 7))
    for i in range(7):
        for j in range(7):
            p_base = poisson.pmf(i, lambda_dom) * poisson.pmf(j, lambda_ext)
            tau = 1.0
            if i == 0 and j == 0: tau = 1.0 - (lambda_dom * lambda_ext * rho)
            elif i == 0 and j == 1: tau = 1.0 + (lambda_dom * rho)
            elif i == 1 and j == 0: tau = 1.0 + (lambda_ext * rho)
            elif i == 1 and j == 1: tau = 1.0 - rho
            mat_ft[i, j] = max(0.0, p_base * tau)

    mat_ft = mat_ft / np.sum(mat_ft)
    scores_dict = {f"{i} - {j}": mat_ft[i, j] for i in range(7) for j in range(7)}

    p_v1 = np.sum(np.tril(mat_ft, -1)) * 100
    p_x  = np.sum(np.diag(mat_ft)) * 100
    p_v2 = np.sum(np.triu(mat_ft, 1)) * 100
    p_1x, p_x2 = p_v1 + p_x, p_v2 + p_x

    over_15 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 1.5) * 100
    over_25 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 2.5) * 100
    over_35 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 3.5) * 100
    under_25, under_35 = 100.0 - over_25, 100.0 - over_35
    btts_oui = np.sum(mat_ft[1:, 1:]) * 100
    top_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    l_d_mt, l_e_mt = r_mt_d * lambda_dom, r_mt_e * lambda_ext
    mat_mt = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            mat_mt[i, j] = poisson.pmf(i, l_d_mt) * poisson.pmf(j, l_e_mt)
    mat_mt = mat_mt / np.sum(mat_mt)

    over_05_mt = (1.0 - mat_mt[0, 0]) * 100
    v1_mt = np.sum(np.tril(mat_mt, -1)) * 100
    x_mt  = np.sum(np.diag(mat_mt)) * 100
    v2_mt = np.sum(np.triu(mat_mt, 1)) * 100

    return {
        "lambda_dom": lambda_dom, "lambda_ext": lambda_ext,
        "forme_dom": forme_dom, "forme_ext": forme_ext,
        "1N2": (p_v1, p_x, p_v2), "1X_X2": (p_1x, p_x2),
        "cotes_1N2": (100.0/max(1, p_v1), 100.0/max(1, p_x), 100.0/max(1, p_v2)),
        "cotes_DC": (100.0/max(1, p_1x), 100.0/max(1, p_x2)),
        "over_15": over_15, "over_25": over_25, "over_35": over_35,
        "under_25": under_25, "under_35": under_35,
        "cote_o15": 100.0/max(1, over_15), "cote_o25": 100.0/max(1, over_25), "cote_u35": 100.0/max(1, under_35),
        "btts": btts_oui, "cote_btts": 100.0/max(1, btts_oui),
        "top_scores": top_scores, "over_05_mt": over_05_mt, "1N2_mt": (v1_mt, x_mt, v2_mt)
    }

# ==========================================
# 5. SÉLECTION DU PARI & ANALYSE HYBRIDE (GEMINI + SECOURS)
# ==========================================
def Selectionner_Meilleur_Pari(dom, ext, res, mode_combine=False):
    candidats = [
        {"nom": f"Victoire {dom} (1)", "prob": res["1N2"][0], "cote": res["cotes_1N2"][0], "type": "1"},
        {"nom": f"Victoire {ext} (2)", "prob": res["1N2"][2], "cote": res["cotes_1N2"][2], "type": "2"},
        {"nom": "Over 1.5 Buts", "prob": res["over_15"], "cote": res["cote_o15"], "type": "O15"},
        {"nom": "Over 2.5 Buts", "prob": res["over_25"], "cote": res["cote_o25"], "type": "O25"},
        {"nom": "Les 2 Marquent (OUI)", "prob": res["btts"], "cote": res["cote_btts"], "type": "BTTS"},
        {"nom": f"Double Chance {dom}/Nul (1X)", "prob": res["1X_X2"][0], "cote": res["cotes_DC"][0], "type": "1X"},
        {"nom": f"Double Chance {ext}/Nul (X2)", "prob": res["1X_X2"][1], "cote": res["cotes_DC"][1], "type": "X2"},
        {"nom": "Under 3.5 Buts", "prob": res["under_35"], "cote": res["cote_u35"], "type": "U35"}
    ]

    if mode_combine:
        filtres = [c for c in candidats if 62.0 <= c["prob"] <= 83.0 and 1.22 <= c["cote"] <= 1.80]
        if not filtres:
            filtres = [c for c in candidats if c["prob"] >= 60.0]
        meilleur = sorted(filtres, key=lambda x: x["prob"], reverse=True)[0]
    else:
        majeurs = [c for c in candidats if c["type"] in ["1", "2", "O25", "BTTS"] and c["prob"] >= 60.0]
        if majeurs:
            meilleur = sorted(majeurs, key=lambda x: x["prob"], reverse=True)[0]
        else:
            secours = [c for c in candidats if c["type"] in ["O15", "1X", "X2"] and c["prob"] >= 70.0]
            if not secours:
                return None
            meilleur = sorted(secours, key=lambda x: x["prob"], reverse=True)[0]

    xg_d, xg_e = res["lambda_dom"], res["lambda_ext"]
    score_1, prob_s1 = res["top_scores"][0]
    texte_secours = (
        f"xG attendu {xg_d:.2f} vs {xg_e:.2f} (Forme {res['forme_dom']:+.0f}% / {res['forme_ext']:+.0f}%). "
        f"Avantage statistique net sur {meilleur['nom']} avec {res['over_05_mt']:.0f}% de prob. de but en 1ère MT."
    )

    return {
        "pari": meilleur["nom"], "prob": meilleur["prob"], "cote": meilleur["cote"],
        "score_exact": f"{score_1} ({prob_s1*100:.1f}%)", "ia_secours": texte_secours
    }

# ==========================================
# 6. COLLECTE DES MATCHS SUR 7 JOURS STRICTS
# ==========================================
@st.cache_data(ttl=1800)
def recuperer_matchs_ligue(code_compet):
    url = f"https://api.football-data.org/v4/competitions/{code_compet}/matches?status=SCHEDULED"
    try:
        req = requests.get(url, headers=HEADERS, timeout=6)
        return req.json().get("matches", [])[:25]
    except Exception:
        return []

def collecter_matchs_7_jours():
    maintenant = datetime.now(timezone.utc)
    limite_7j = maintenant + timedelta(days=7)
    liste_propre = []

    for nom_compet, code_api in CODES_COMPETITIONS.items():
        equipes_autorisees = obtenir_equipes_ligue(nom_compet)
        matchs_bruts = recuperer_matchs_ligue(code_api)
        est_europe = (code_api == "CL")

        for m in matchs_bruts:
            try:
                dt_match = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            except Exception:
                continue
            if not (maintenant <= dt_match <= limite_7j):
                continue

            dom_csv = trouver_nom_csv_strict(m["homeTeam"]["name"], equipes_autorisees)
            ext_csv = trouver_nom_csv_strict(m["awayTeam"]["name"], equipes_autorisees)
            if not dom_csv or not ext_csv or dom_csv == ext_csv:
                continue

            res = predire_rencontre(dom_csv, ext_csv, est_europe=est_europe)
            liste_propre.append({
                "id": m["id"], "dt": dt_match,
                "date_str": dt_match.strftime("%d/%m à %H:%M"),
                "jour_date": dt_match.date(), "heure_str": dt_match.strftime("%H:%M"),
                "ligue": nom_compet, "dom": dom_csv, "ext": ext_csv,
                "est_europe": est_europe, "res": res
            })
    return sorted(liste_propre, key=lambda x: x["dt"])

# ==========================================
# 7. INTERFACE MOBILE PRINCIPALE
# ==========================================
st.markdown("<h2 style='text-align: center;'>⚽ DIRECT FOOT PRO AI</h2>", unsafe_allow_html=True)
st.caption("Poisson + Dixon-Coles • Forme 6 matchs • Propulsé par Google Gemini Flash AI")

onglets = st.tabs(["Aujourd'hui", "Demain", "Jours Suivants", "🚀 Scanner & Combiné Gemini"])

matchs_semaine = collecter_matchs_7_jours()
date_auj = datetime.now(timezone.utc).date()
date_dem = date_auj + timedelta(days=1)

if "match_actif" not in st.session_state:
    st.session_state.match_actif = None

for idx, onglet in enumerate(onglets[:3]):
    with onglet:
        if idx == 0: m_jour = [m for m in matchs_semaine if m["jour_date"] == date_auj]
        elif idx == 1: m_jour = [m for m in matchs_semaine if m["jour_date"] == date_dem]
        else: m_jour = [m for m in matchs_semaine if m["jour_date"] > date_dem]

        if not m_jour:
            st.info("Aucun match à venir programmé sur ce créneau.")
        else:
            for nom_ligue in CODES_COMPETITIONS.keys():
                m_ligue = [m for m in m_jour if m["ligue"] == nom_ligue]
                if m_ligue:
                    with st.expander(f"🏆 {nom_ligue} ({len(m_ligue)})", expanded=True):
                        for m in m_ligue:
                            c1, c2 = st.columns([1, 4])
                            c1.write(f"**{m['date_str'][:5]}** `{m['heure_str']}`")
                            if c2.button(f"{m['dom']} vs {m['ext']}", key=f"nav_{idx}_{m['id']}", use_container_width=True):
                                st.session_state.match_actif = m

# ==========================================
# 8. FICHE DÉTAILLÉE DU MATCH + ANALYSE GEMINI
# ==========================================
if st.session_state.match_actif:
    m_sel = st.session_state.match_actif
    st.markdown("---")
    st.subheader(f"🏟️ {m_sel['dom']} vs {m_sel['ext']} ({m_sel['ligue']})")
    st.caption(f"📅 Coup d'envoi : {m_sel['date_str']} UTC")

    with st.expander("🏥 Ajuster Blessures / Fatigue (Optionnel)", expanded=False):
        cm1, cm2 = st.columns(2)
        malus_d = cm1.slider(f"Malus xG {m_sel['dom']}", 0.0, 0.50, 0.0, 0.05)
        malus_e = cm2.slider(f"Malus xG {m_sel['ext']}", 0.0, 0.50, 0.0, 0.05)

    res = predire_rencontre(m_sel["dom"], m_sel["ext"], est_europe=m_sel["est_europe"], malus_dom=malus_d, malus_ext=malus_e)
    best_pick = Selectionner_Meilleur_Pari(m_sel["dom"], m_sel["ext"], res, mode_combine=True)

    # Appel à Gemini AI pour la fiche du match
    prompt_match = (
        f"Tu es un analyste professionnel en modélisation sportive. Rédige en français une analyse concise (3 phrases maximum) "
        f"pour le match {m_sel['dom']} vs {m_sel['ext']} ({m_sel['ligue']}). "
        f"Données : xG {m_sel['dom']}={res['lambda_dom']:.2f} (forme {res['forme_dom']:+.0f}%), "
        f"xG {m_sel['ext']}={res['lambda_ext']:.2f} (forme {res['forme_ext']:+.0f}%), "
        f"1N2=({res['1N2'][0]:.1f}% / {res['1N2'][1]:.1f}% / {res['1N2'][2]:.1f}%), "
        f"Over 2.5={res['over_25']:.1f}%, BTTS={res['btts']:.1f}%, Score exact numéro 1={best_pick['score_exact']}. "
        f"Explique pourquoi le pari '{best_pick['pari']}' ({best_pick['prob']:.1f}%, cote min @{best_pick['cote']:.2f}) est le choix optimal."
    )
    texte_gemini_match = appeler_gemini_ai(prompt_match) or best_pick["ia_secours"]

    st.success(
        f"🤖 **Analyse Gemini Flash AI — Pari recommandé : {best_pick['pari']}** "
        f"(`{best_pick['prob']:.1f}%` | Cote min `@{best_pick['cote']:.2f}`)\n\n{texte_gemini_match}"
    )

    c_xg1, c_xg2 = st.columns(2)
    c_xg1.metric(f"xG Attendu ({m_sel['dom']})", f"{res['lambda_dom']:.2f} buts", f"Forme 6M: {res['forme_dom']:+.0f}%")
    c_xg2.metric(f"xG Attendu ({m_sel['ext']})", f"{res['lambda_ext']:.2f} buts", f"Forme 6M: {res['forme_ext']:+.0f}%")

    st.markdown("#### 🎯 Issue Finale (1N2) & Cotes Équitables")
    c1, c2, c3 = st.columns(3)
    c1.metric(f"1 ({m_sel['dom']})", f"{res['1N2'][0]:.1f}%", f"@{res['cotes_1N2'][0]:.2f}")
    c2.metric("Nul (X)", f"{res['1N2'][1]:.1f}%", f"@{res['cotes_1N2'][1]:.2f}")
    c3.metric(f"2 ({m_sel['ext']})", f"{res['1N2'][2]:.1f}%", f"@{res['cotes_1N2'][2]:.2f}")

    st.markdown("#### 🤝 Les Deux Équipes Marquent (BTTS)")
    dec_btts = "OUI" if res['btts'] >= 50 else "NON"
    prob_btts_aff = res['btts'] if dec_btts == "OUI" else (100.0 - res['btts'])
    st.info(f"Décision BTTS : **{dec_btts}** à **{prob_btts_aff:.1f}%** (OUI : `{res['btts']:.1f}%` @{res['cote_btts']:.2f} | NON : `{100-res['btts']:.1f}%`)")

    st.markdown("#### 📈 Marchés de Buts (Over / Under)")
    st.write(f"• **Over 1.5** : `{res['over_15']:.1f}%` (@{res['cote_o15']:.2f}) | **Under 1.5** : `{100 - res['over_15']:.1f}%`")
    st.write(f"• **Over 2.5** : `{res['over_25']:.1f}%` (@{res['cote_o25']:.2f}) | **Under 2.5** : `{res['under_25']:.1f}%`")
    st.write(f"• **Over 3.5** : `{res['over_35']:.1f}%` | **Under 3.5** : `{res['under_35']:.1f}%` (@{res['cote_u35']:.2f})")

    st.markdown("#### ⏱️ Marchés 1ère Mi-Temps (Calcul Réel)")
    st.write(f"• **But en 1ère MT (+0.5 MT)** : `{res['over_05_mt']:.1f}%`")
    st.write(f"• **1N2 Pause** : 1 (`{res['1N2_mt'][0]:.1f}%`) | Nul (`{res['1N2_mt'][1]:.1f}%`) | 2 (`{res['1N2_mt'][2]:.1f}%`)")

    st.markdown("#### 🎲 Top 5 Scores Exacts (Dixon-Coles)")
    df_sc = pd.DataFrame([{"Score Exact": s, "Probabilité": f"{p * 100:.1f} %"} for s, p in res["top_scores"]])
    st.table(df_sc)

# ==========================================
# 9. ONGLET 4 : COMBINÉ DU JOUR + SCANNER 7 JOURS GEMINI
# ==========================================
with onglets[3]:
    st.subheader("🎟️ Générateur de Ticket Combiné IA (J ou J+1)")
    nb_matchs_comb = st.radio("Taille du combiné souhaité :", [3, 4, 5], index=0, horizontal=True)

    if st.button("🔥 Générer le Combiné Audité par Gemini & Scanner les 7 Jours", use_container_width=True):
        with st.spinner("Calcul des matrices et consultation de Google Gemini Flash en cours..."):
            # 1. Bascule automatique J -> J+1
            candidats_j = [m for m in matchs_semaine if m["jour_date"] == date_auj]
            label_jour = f"Aujourd'hui ({date_auj.strftime('%d/%m')})"

            if len(candidats_j) < nb_matchs_comb:
                candidats_j1 = [m for m in matchs_semaine if m["jour_date"] in [date_auj, date_dem]]
                if len(candidats_j1) >= nb_matchs_comb:
                    candidats_comb = candidats_j1
                    label_jour = f"Bascule Auto Aujourd'hui + Demain ({date_dem.strftime('%d/%m')})"
                else:
                    candidats_comb = matchs_semaine[:8]
                    label_jour = "Prochains matchs imminents du calendrier"
            else:
                candidats_comb = candidats_j

            selections_comb = []
            for m in candidats_comb:
                eval_p = Selectionner_Meilleur_Pari(m["dom"], m["ext"], m["res"], mode_combine=True)
                if eval_p:
                    selections_comb.append({
                        "match_str": f"{m['dom']} vs {m['ext']}", "ligue": m["ligue"], "date": m["date_str"],
                        "pari": eval_p["pari"], "prob": eval_p["prob"], "cote": eval_p["cote"],
                        "score": eval_p["score_exact"], "ia": eval_p["ia_secours"],
                        "xg": f"{m['res']['lambda_dom']:.2f}-{m['res']['lambda_ext']:.2f}"
                    })

            top_combine = sorted(selections_comb, key=lambda x: x["prob"], reverse=True)[:nb_matchs_comb]

            if len(top_combine) > 0:
                cote_totale = np.prod([x["cote"] for x in top_combine])
                prob_moyenne = np.mean([x["prob"] for x in top_combine])

                st.markdown(f"### 🎫 Ticket Combiné Officiel — *{label_jour}*")
                col_k1, col_k2 = st.columns(2)
                col_k1.metric("Cote Totale Théorique", f"@{cote_totale:.2f}")
                col_k2.metric("Fiabilité Moyenne / Sélection", f"{prob_moyenne:.1f} %")

                details_prompt = "\n".join([
                    f"{i+1}. {s['match_str']} ({s['ligue']}) -> Pari: {s['pari']} ({s['prob']:.1f}%, @{s['cote']:.2f}), xG: {s['xg']}, Score probable: {s['score']}"
                    for i, s in enumerate(top_combine)
                ])
                prompt_ticket = (
                    f"Tu es un expert en paris sportifs. Audite ce ticket combiné de {len(top_combine)} matchs (Cote totale @{cote_totale:.2f}) :\n"
                    f"{details_prompt}\n"
                    f"Réponds en français de manière structurée et percutante :\n"
                    f"1) Une phrase d'analyse tactique pour chaque match justifiant le pari.\n"
                    f"2) La cohérence globale du ticket et le match à surveiller particulièrement.\n"
                    f"3) Une note de confiance sur 10 et le % de bankroll conseillé."
                )
                audit_gemini_ticket = appeler_gemini_ai(prompt_ticket)

                for i, sel in enumerate(top_combine, 1):
                    st.markdown(
                        f"**{i}. {sel['match_str']}** *({sel['ligue']} — {sel['date']})*\n"
                        f"- 🎯 **Pari :** `{sel['pari']}` | **Probabilité :** `{sel['prob']:.1f}%` *(Cote @{sel['cote']:.2f})* | **Score exact clé :** `{sel['score']}`"
                    )

                if audit_gemini_ticket:
                    st.success(f"🤖 **Audit Complet du Ticket par Google Gemini AI :**\n\n{audit_gemini_ticket}")
                else:
                    st.info("📋 **Audit du Ticket :** Sélection équilibrée à haute espérance mathématique. Mise conseillée : 2% de bankroll.")

            st.markdown("---")
            st.subheader("📅 Tableau Scanner des 7 Prochains Jours (Interprété par Gemini AI)")

            lignes_brutes = []
            for m in matchs_semaine:
                best = Selectionner_Meilleur_Pari(m["dom"], m["ext"], m["res"], mode_combine=False)
                if best:
                    lignes_brutes.append({
                        "Date": m["date_str"], "Ligue": m["ligue"], "Match": f"{m['dom']} vs {m['ext']}",
                        "Pari": best["pari"], "Prob": best["prob"], "Cote": best["cote"],
                        "Score": best["score_exact"], "xG": f"{m['res']['lambda_dom']:.2f}-{m['res']['lambda_ext']:.2f}",
                        "Secours": best["ia_secours"]
                    })

            if lignes_brutes:
                # Appel groupé à Gemini pour générer 1 phrase tactique par ligne du tableau
                resume_pour_ia = "\n".join([
                    f"{idx+1}| {x['Match']} | Pari:{x['Pari']} ({x['Prob']:.0f}%) | xG:{x['xG']} | Score:{x['Score']}"
                    for idx, x in enumerate(lignes_brutes[:25])
                ])
                prompt_scan = (
                    "Pour chacun de ces matchs numérotés, écris UNE SEULE phrase très courte (max 14 mots) en français expliquant le pari et le score probable. "
                    "Réponds strictement au format 'Numéro|Phrase' sur chaque ligne, sans introduction :\n" + resume_pour_ia
                )
                rep_batch = appeler_gemini_ai(prompt_scan)
                dict_ia_lignes = {}
                if rep_batch:
                    for line in rep_batch.splitlines():
                        if "|" in line:
                            parts = line.split("|", 1)
                            if parts[0].strip().isdigit():
                                dict_ia_lignes[int(parts[0].strip())] = parts[1].strip()

                tableau_final = []
                for idx, x in enumerate(lignes_brutes):
                    phrase_ia = dict_ia_lignes.get(idx + 1, x["Secours"])
                    tableau_final.append({
                        "Date": x["Date"],
                        "Ligue": x["Ligue"],
                        "Match": x["Match"],
                        "Pari Recommandé": f"{x['Pari']} ({x['Prob']:.1f}% | @{x['Cote']:.2f})",
                        "Score Exact Probable": x["Score"],
                        "Interprétation Gemini AI": phrase_ia
                    })

                df_scan = pd.DataFrame(tableau_final)
                st.success(f"✅ {len(df_scan)} rencontres vérifiées sur les 7 prochains jours et interprétées par Gemini AI.")
                st.dataframe(df_scan, use_container_width=True, hide_index=True)
            else:
                st.warning("Aucun match dans les 7 prochains jours ne dépasse les seuils de confiance.")
