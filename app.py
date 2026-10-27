import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta, timezone

# ==========================================
# CONFIGURATION & CSS PLEIN ÉCRAN
# ==========================================
st.set_page_config(page_title="Direct Foot Pro AI", page_icon="⚽", layout="wide")

st.markdown("""
    <style>
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        padding-left: 1.5rem !important;
        padding-right: 1.5rem !important;
        max-width: 1600px !important;
    }
    </style>
""", unsafe_allow_html=True)

API_TOKEN = "154e2f8f245f4d07b64977f1bf2d9ec9"
HEADERS = {"X-Auth-Token": API_TOKEN}
GEMINI_API_KEY = "AQ.Ab8RN6L5dMyifo8wWnczhTj4dpkYg68sRGYYIMFDvJNTeW2kpA"

CODES_COMPETITIONS = {
    "UEFA Champions League": "CL", "Premier League": "PL", "La Liga": "PD",
    "Serie A": "SA", "Bundesliga": "BL1", "Ligue 1": "FL1"
}

# ==========================================
# 1. DICTIONNAIRE EXACT & ANTI-ERREURS
# ==========================================
CORRESPONDANCES_NOMS = {
    "Paris Saint-Germain FC": "Paris SG", "Olympique de Marseille": "Marseille", "Olympique Lyonnais": "Lyon",
    "AS Monaco FC": "Monaco", "LOSC Lille": "Lille", "OGC Nice": "Nice", "Stade Rennais FC 1904": "Rennes",
    "RC Lens": "Lens", "Stade Brestois 29": "Brest", "RC Strasbourg Alsace": "Strasbourg", "FC Nantes": "Nantes",
    "Montpellier HSC": "Montpellier", "Toulouse FC": "Toulouse", "Stade de Reims": "Reims", "Le Havre AC": "Le Havre",
    "AJ Auxerre": "Auxerre", "Angers SCO": "Angers", "AS Saint-Étienne": "St Etienne",
    "Arsenal FC": "Arsenal", "Aston Villa FC": "Aston Villa", "AFC Bournemouth": "Bournemouth", "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton", "Chelsea FC": "Chelsea", "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton", "Fulham FC": "Fulham", "Ipswich Town FC": "Ipswich", "Leicester City FC": "Leicester",
    "Liverpool FC": "Liverpool", "Manchester City FC": "Man City", "Manchester United FC": "Man United",
    "Newcastle United FC": "Newcastle", "Nottingham Forest FC": "Nott'm Forest", "Southampton FC": "Southampton",
    "Tottenham Hotspur FC": "Tottenham", "West Ham United FC": "West Ham", "Wolverhampton Wanderers FC": "Wolves",
    "FC Barcelona": "Barcelona", "Real Madrid CF": "Real Madrid", "Club Atlético de Madrid": "Ath Madrid",
    "Athletic Club": "Ath Bilbao", "Real Sociedad de Fútbol": "Sociedad", "Real Betis Balompié": "Betis",
    "Villarreal CF": "Villarreal", "Valencia CF": "Valencia", "Sevilla FC": "Sevilla", "Girona FC": "Girona",
    "RC Celta de Vigo": "Celta", "CA Osasuna": "Osasuna", "Getafe CF": "Getafe", "Rayo Vallecano de Madrid": "Vallecano",
    "RCD Mallorca": "Mallorca", "Deportivo Alavés": "Alaves", "UD Las Palmas": "Las Palmas", "CD Leganés": "Leganes",
    "RCD Espanyol de Barcelona": "Espanyol", "Real Valladolid CF": "Valladolid",
    "FC Internazionale Milano": "Inter", "AC Milan": "Milan", "Juventus FC": "Juventus", "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna", "AS Roma": "Roma", "SS Lazio": "Lazio", "ACF Fiorentina": "Fiorentina",
    "SSC Napoli": "Napoli", "Torino FC": "Torino", "Genoa CFC": "Genoa", "AC Monza": "Monza",
    "Hellas Verona FC": "Verona", "US Lecce": "Lecce", "Udinese Calcio": "Udinese", "Cagliari Calcio": "Cagliari",
    "Empoli FC": "Empoli", "Parma Calcio 1913": "Parma", "Como 1907": "Como", "Venezia FC": "Venezia",
    "Bayer 04 Leverkusen": "Leverkusen", "FC Bayern München": "Bayern Munich", "VfB Stuttgart": "Stuttgart",
    "RB Leipzig": "RB Leipzig", "Borussia Dortmund": "Dortmund", "Eintracht Frankfurt": "Ein Frankfurt",
    "TSG 1899 Hoffenheim": "Hoffenheim", "1. FC Heidenheim 1846": "Heidenheim", "SV Werder Bremen": "Werder Bremen",
    "SC Freiburg": "Freiburg", "FC Augsburg": "Augsburg", "VfL Wolfsburg": "Wolfsburg", "1. FSV Mainz 05": "Mainz",
    "Borussia Mönchengladbach": "M'gladbach", "1. FC Union Berlin": "Union Berlin", "VfL Bochum 1848": "Bochum",
    "FC St. Pauli 1910": "St Pauli", "Holstein Kiel": "Holstein Kiel"
}

def trouver_nom_csv_strict(nom_api, equipes_autorisees):
    if not nom_api or len(equipes_autorisees) == 0: return None
    if nom_api in CORRESPONDANCES_NOMS:
        c = CORRESPONDANCES_NOMS[nom_api]
        if c in equipes_autorisees: return c

    mots = ["FC", "CF", "AS", "SSC", "AC", "SC", "SV", "RC", "RCD", "UD", "CD", "AFC", "1909", "1904", "1899", "1846", "1848", "1910", "1913", "1907", "04", "05"]
    nom_net = nom_api
    for m in mots: nom_net = nom_net.replace(f" {m} ", " ").replace(f" {m}", "").replace(f"{m} ", "")
    nom_net = nom_net.strip()

    for eq in equipes_autorisees:
        if eq.lower() == nom_net.lower() or eq.lower() == nom_api.lower(): return eq
    for eq in equipes_autorisees:
        if len(eq) >= 4 and (eq.lower() in nom_net.lower() or nom_net.lower() in eq.lower()): return eq

    p = difflib.get_close_matches(nom_net, equipes_autorisees, n=1, cutoff=0.58)
    return p[0] if p else None

# ==========================================
# 2. CONNECTEUR GEMINI RAPIDE
# ==========================================
def appeler_gemini_ai(prompt_texte):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt_texte}]}],
        "generationConfig": {"temperature": 0.4, "maxOutputTokens": 1500}
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code == 200: return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            url_alt = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
            r_alt = requests.post(url_alt, headers=headers, json=payload, timeout=60)
            if r_alt.status_code == 200: return r_alt.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            return f"Notice API ({r.status_code}) : {r.text[:120]}"
    except Exception: return f"Erreur de délai réseau (Timeout). Veuillez reformuler plus simplement."

# ==========================================
# 3. STATISTIQUES & MODÈLE (MI-TEMPS INTÉGRÉES)
# ==========================================
@st.cache_data
def charger_historique(): return pd.read_csv("historique_europe_complet.csv")

df_total = charger_historique()
TOUTES_EQUIPES_CSV = sorted(list(set(df_total['Domicile'].unique()).union(set(df_total['Exterieur'].unique()))))

def obtenir_equipes_ligue(nom_compet):
    if nom_compet in df_total['Championnat'].values:
        df_l = df_total[df_total['Championnat'] == nom_compet]
        return sorted(list(set(df_l['Domicile'].unique()).union(set(df_l['Exterieur'].unique()))))
    return TOUTES_EQUIPES_CSV

def calculer_forces_completes(equipe):
    matchs = df_total[(df_total['Domicile'] == equipe) | (df_total['Exterieur'] == equipe)]
    if len(matchs) == 0: return 1.0, 1.0, 1.0, 1.0, 1.50, 1.20, 0.44, 0.44, 0.0

    df_ligue = df_total[df_total['Championnat'] == matchs['Championnat'].iloc[0]]
    moy_dom, moy_ext = df_ligue['Buts_Dom'].mean(), df_ligue['Buts_Ext'].mean()

    m_dom, m_ext = df_ligue[df_ligue['Domicile'] == equipe], df_ligue[df_ligue['Exterieur'] == equipe]
    
    att_d_h = (m_dom['Buts_Dom'].mean() / moy_dom) if len(m_dom) > 0 else 1.0
    def_d_h = (m_dom['Buts_Ext'].mean() / moy_ext) if len(m_dom) > 0 else 1.0
    att_e_h = (m_ext['Buts_Ext'].mean() / moy_ext) if len(m_ext) > 0 else 1.0
    def_e_h = (m_ext['Buts_Dom'].mean() / moy_dom) if len(m_ext) > 0 else 1.0

    m_dom_rec, m_ext_rec = m_dom.tail(6), m_ext.tail(6)
    att_d_f = (m_dom_rec['Buts_Dom'].mean() / moy_dom) if len(m_dom_rec) > 0 else att_d_h
    def_d_f = (m_dom_rec['Buts_Ext'].mean() / moy_ext) if len(m_dom_rec) > 0 else def_d_h
    att_e_f = (m_ext_rec['Buts_Ext'].mean() / moy_ext) if len(m_ext_rec) > 0 else att_e_h
    def_e_f = (m_ext_rec['Buts_Dom'].mean() / moy_dom) if len(m_ext_rec) > 0 else def_e_h

    att_d, def_d = (0.60 * att_d_h) + (0.40 * att_d_f), (0.60 * def_d_h) + (0.40 * def_d_f)
    att_e, def_e = (0.60 * att_e_h) + (0.40 * att_e_f), (0.60 * def_e_h) + (0.40 * def_e_f)

    indice_forme = ((att_d_f + att_e_f) / max(0.5, (att_d_h + att_e_h)) - 1.0) * 100
    r_mt_d, r_mt_e = 0.44, 0.44
    if 'Buts_Dom_MT' in df_ligue.columns and 'Buts_Ext_MT' in df_ligue.columns:
        tot_d, tot_e = m_dom['Buts_Dom'].sum(), m_ext['Buts_Ext'].sum()
        if tot_d > 5: r_mt_d = np.clip(m_dom['Buts_Dom_MT'].sum() / tot_d, 0.30, 0.60)
        if tot_e > 5: r_mt_e = np.clip(m_ext['Buts_Ext_MT'].sum() / tot_e, 0.30, 0.60)

    return att_d, def_d, att_e, def_e, moy_dom, moy_ext, r_mt_d, r_mt_e, indice_forme

def predire_rencontre(equipe_dom, equipe_ext, est_europe=False, malus_dom=0.0, malus_ext=0.0):
    att_d, def_d, _, _, moy_dom_nat, _, r_mt_d, _, forme_dom = calculer_forces_completes(equipe_dom)
    _, _, att_e, def_e, _, moy_ext_nat, _, r_mt_e, forme_ext = calculer_forces_completes(equipe_ext)

    base_dom, base_ext = (1.62, 1.32) if est_europe else (moy_dom_nat, moy_ext_nat)
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
    top_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:15]

    p_v1, p_x, p_v2 = np.sum(np.tril(mat_ft, -1))*100, np.sum(np.diag(mat_ft))*100, np.sum(np.triu(mat_ft, 1))*100
    p_1x, p_x2 = p_v1 + p_x, p_v2 + p_x

    over_15 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 1.5) * 100
    over_25 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 2.5) * 100
    under_25 = 100.0 - over_25
    btts_oui = np.sum(mat_ft[1:, 1:]) * 100
    btts_non = 100.0 - btts_oui
    
    l_d_mt1, l_e_mt1 = r_mt_d * lambda_dom, r_mt_e * lambda_ext
    l_d_mt2, l_e_mt2 = (1 - r_mt_d) * lambda_dom, (1 - r_mt_e) * lambda_ext
    p0_mt1 = np.exp(-l_d_mt1 - l_e_mt1)
    p1_mt1 = (l_d_mt1 + l_e_mt1) * p0_mt1
    over_15_mt1 = (1.0 - p0_mt1 - p1_mt1) * 100
    p0_mt2 = np.exp(-l_d_mt2 - l_e_mt2)
    p1_mt2 = (l_d_mt2 + l_e_mt2) * p0_mt2
    over_15_mt2 = (1.0 - p0_mt2 - p1_mt2) * 100
    but_chaque_mt = (1.0 - p0_mt1) * (1.0 - p0_mt2) * 100

    return {
        "lambda_dom": lambda_dom, "lambda_ext": lambda_ext,
        "forme_dom": forme_dom, "forme_ext": forme_ext,
        "1N2": (p_v1, p_x, p_v2), "1X_X2": (p_1x, p_x2),
        "cotes_1N2": (100.0/max(1, p_v1), 100.0/max(1, p_x), 100.0/max(1, p_v2)),
        "cotes_DC": (100.0/max(1, p_1x), 100.0/max(1, p_x2)),
        "over_15": over_15, "over_25": over_25, "under_25": under_25,
        "cote_o15": 100.0/max(1, over_15), "cote_o25": 100.0/max(1, over_25), "cote_u25": 100.0/max(1, under_25),
        "btts": btts_oui, "btts_non": btts_non,
        "cote_btts": 100.0/max(1, btts_oui), "cote_btts_non": 100.0/max(1, btts_non),
        "over_15_mt1": over_15_mt1, "cote_o15_mt1": 100.0/max(1, over_15_mt1),
        "over_15_mt2": over_15_mt2, "cote_o15_mt2": 100.0/max(1, over_15_mt2),
        "but_chaque_mt": but_chaque_mt, "cote_bcm": 100.0/max(1, but_chaque_mt),
        "top_scores": top_scores
    }

def analyser_paris_match(dom, ext, res):
    return [
        {"pari": f"Victoire {dom} (1)", "prob": res["1N2"][0], "cote": res["cotes_1N2"][0], "type": "1"},
        {"pari": f"Victoire {ext} (2)", "prob": res["1N2"][2], "cote": res["cotes_1N2"][2], "type": "2"},
        {"pari": "Match Nul (X)", "prob": res["1N2"][1], "cote": res["cotes_1N2"][1], "type": "X"},
        {"pari": "Over 2.5 Buts", "prob": res["over_25"], "cote": res["cote_o25"], "type": "O25"},
        {"pari": "Under 2.5 Buts", "prob": res["under_25"], "cote": res["cote_u25"], "type": "U25"},
        {"pari": "Les 2 Marquent (Oui)", "prob": res["btts"], "cote": res["cote_btts"], "type": "BTTS_O"},
        {"pari": "Les 2 Marquent (Non)", "prob": res["btts_non"], "cote": res["cote_btts_non"], "type": "BTTS_N"},
        {"pari": "But dans chaque Mi-Temps", "prob": res["but_chaque_mt"], "cote": res["cote_bcm"], "type": "BCM"},
        {"pari": "Over 1.5 en 1ère MT", "prob": res["over_15_mt1"], "cote": res["cote_o15_mt1"], "type": "O15_MT1"},
        {"pari": "Over 1.5 en 2ème MT", "prob": res["over_15_mt2"], "cote": res["cote_o15_mt2"], "type": "O15_MT2"}
    ]

def selectionner_meilleur_pari_value(res, tous_paris):
    candidats = [p for p in tous_paris if p["cote"] >= 1.40]
    if not candidats: candidats = tous_paris
    
    score_1_brut = res["top_scores"][0][0]
    xg_total = res["lambda_dom"] + res["lambda_ext"]
    xg_diff = res["lambda_dom"] - res["lambda_ext"]

    if score_1_brut in ["0 - 0", "1 - 0", "0 - 1", "1 - 1"]:
        if xg_total < 2.5: candidats = [p for p in candidats if p["type"] not in ["O25", "O15_MT1", "O15_MT2", "BCM", "BTTS_O"]]
    else:
        if xg_total > 2.5: candidats = [p for p in candidats if p["type"] not in ["U25", "BTTS_N", "X"]]
            
    if not candidats: candidats = tous_paris
    top_pari = sorted(candidats, key=lambda x: x["prob"], reverse=True)[0]

    t_pari = top_pari["type"]
    meilleur_score = res["top_scores"][0]
    meilleur_poids = -1.0
    
    for sc_str, sc_prob in res["top_scores"]:
        s_d, s_e = int(sc_str.split(" - ")[0]), int(sc_str.split(" - ")[1])
        match_ok = False
        
        if t_pari == "1" and s_d > s_e: match_ok = True
        elif t_pari == "2" and s_d < s_e: match_ok = True
        elif t_pari == "X" and s_d == s_e: match_ok = True
        elif t_pari in ["O25", "BCM"] and (s_d + s_e) >= 3: match_ok = True
        elif t_pari in ["O15", "O15_MT1", "O15_MT2"] and (s_d + s_e) >= 2: match_ok = True
        elif t_pari == "U25" and (s_d + s_e) <= 2: match_ok = True
        elif t_pari == "BTTS_O" and s_d > 0 and s_e > 0: match_ok = True
        elif t_pari == "BTTS_N" and (s_d == 0 or s_e == 0): match_ok = True
        
        if match_ok:
            poids = sc_prob * 100 
            if abs(xg_diff) > 1.2 and abs(s_d - s_e) >= 2: poids += 4.0
            if xg_total >= 3.0 and (s_d + s_e) >= 3: poids += 3.0
            if xg_total >= 3.5 and (s_d + s_e) >= 4: poids += 5.0
                
            if poids > meilleur_poids:
                meilleur_poids = poids
                meilleur_score = (sc_str, sc_prob)

    return top_pari, meilleur_score

# ==========================================
# 4. COLLECTE DES MATCHS SUR 7 JOURS
# ==========================================
@st.cache_data(ttl=1800)
def recuperer_matchs_ligue(code_compet):
    url = f"https://api.football-data.org/v4/competitions/{code_compet}/matches?status=SCHEDULED"
    try:
        req = requests.get(url, headers=HEADERS, timeout=6)
        return req.json().get("matches", [])[:25]
    except Exception: return []

def collecter_matchs_7_jours():
    maintenant = datetime.now(timezone.utc)
    limite_7j = maintenant + timedelta(days=7)
    liste_propre, vus = [], set()

    for nom_compet, code_api in CODES_COMPETITIONS.items():
        equipes_autorisees = obtenir_equipes_ligue(nom_compet)
        matchs_bruts = recuperer_matchs_ligue(code_api)
        est_europe = (code_api == "CL")

        for m in matchs_bruts:
            try: dt_match = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            except Exception: continue
            if not (maintenant <= dt_match <= limite_7j): continue

            cle_match = f"{m['homeTeam']['name']}_{dt_match.date()}"
            if cle_match in vus: continue
            vus.add(cle_match)

            dom_csv = trouver_nom_csv_strict(m["homeTeam"]["name"], equipes_autorisees)
            ext_csv = trouver_nom_csv_strict(m["awayTeam"]["name"], equipes_autorisees)
            if not dom_csv or not ext_csv or dom_csv == ext_csv: continue

            res = predire_rencontre(dom_csv, ext_csv, est_europe=est_europe)
            tous_paris = analyser_paris_match(dom_csv, ext_csv, res)

            liste_propre.append({
                "id": m["id"], "dt": dt_match, "date_str": dt_match.strftime("%d/%m à %H:%M"),
                "jour_date": dt_match.date(), "heure_str": dt_match.strftime("%H:%M"),
                "ligue": nom_compet, "dom": dom_csv, "ext": ext_csv,
                "est_europe": est_europe, "res": res, "paris": tous_paris
            })
    return sorted(liste_propre, key=lambda x: x["dt"])

# ==========================================
# 5. INTERFACE MOBILE & DESKTOP (WIDE)
# ==========================================
st.markdown("<h2 style='text-align: center;'>⚽ DIRECT FOOT PRO AI</h2>", unsafe_allow_html=True)

onglets = st.tabs(["Aujourd'hui", "Demain", "Jours Suivants", "🎟️ Tickets IA", "📅 Scanner Value"])
matchs_semaine = collecter_matchs_7_jours()
date_auj = datetime.now(timezone.utc).date()
date_dem = date_auj + timedelta(days=1)

if "match_actif" not in st.session_state: st.session_state.match_actif = None
if "messages_chat" not in st.session_state: st.session_state.messages_chat = []

for idx, onglet in enumerate(onglets[:3]):
    with onglet:
        if idx == 0: m_jour = [m for m in matchs_semaine if m["jour_date"] == date_auj]
        elif idx == 1: m_jour = [m for m in matchs_semaine if m["jour_date"] == date_dem]
        else: m_jour = [m for m in matchs_semaine if m["jour_date"] > date_dem]

        if not m_jour: st.info("Aucun match à venir programmé sur ce créneau.")
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

# Fiche détaillée
if st.session_state.match_actif:
    m_sel = st.session_state.match_actif
    res = predire_rencontre(m_sel["dom"], m_sel["ext"], est_europe=m_sel["est_europe"])
    st.markdown("---")
    st.subheader(f"🏟️ {m_sel['dom']} vs {m_sel['ext']} ({m_sel['ligue']})")
    
    c1, c2, c3 = st.columns(3)
    c1.metric(f"1 ({m_sel['dom']})", f"{res['1N2'][0]:.1f}%", f"@{res['cotes_1N2'][0]:.2f}")
    c2.metric("Nul (X)", f"{res['1N2'][1]:.1f}%", f"@{res['cotes_1N2'][1]:.2f}")
    c3.metric(f"2 ({m_sel['ext']})", f"{res['1N2'][2]:.1f}%", f"@{res['cotes_1N2'][2]:.2f}")
    
    col_a, col_b = st.columns(2)
    col_a.info(f"**BTTS (Oui)**: `{res['btts']:.1f}%` (@{res['cote_btts']:.2f}) | **BTTS (Non)**: `{res['btts_non']:.1f}%`")
    col_b.success(f"**Over 2.5**: `{res['over_25']:.1f}%` (@{res['cote_o25']:.2f}) | **Under 2.5**: `{res['under_25']:.1f}%`")
    st.write(f"⏱️ **But dans chaque MT** : `{res['but_chaque_mt']:.1f}%` (@{res['cote_bcm']:.2f})")
    st.write(f"🎯 **Score le plus probable** : `{res['top_scores'][0][0]}` ({res['top_scores'][0][1]*100:.1f}%)")

# ==========================================
# 6. ONGLET 4 : MULTI-TICKETS & CHAT
# ==========================================
with onglets[3]:
    st.subheader("🎟️ Générateur Multi-Tickets (Cotes @2 à @15+)")

    pool = [m for m in matchs_semaine if m["jour_date"] == date_auj]
    if len(pool) < 4: pool = [m for m in matchs_semaine if m["jour_date"] in [date_auj, date_dem]]
    if len(pool) < 4: pool = matchs_semaine[:12]

    def fabriquer_ticket(min_cote, max_cote, min_prob, max_matchs):
        ticket, vus = [], set()
        for m in pool:
            if m["id"] in vus: continue
            bons_paris = [p for p in m["paris"] if min_cote <= p["cote"] <= max_cote and p["prob"] >= min_prob]
            score_1_b, xg_t = m["res"]["top_scores"][0][0], m["res"]["lambda_dom"] + m["res"]["lambda_ext"]
            if score_1_b in ["0 - 0", "1 - 0", "0 - 1", "1 - 1"]:
                if xg_t < 2.5: bons_paris = [p for p in bons_paris if p["type"] not in ["O25", "O15_MT1", "O15_MT2", "BCM", "BTTS_O"]]
            else:
                if xg_t > 2.5: bons_paris = [p for p in bons_paris if p["type"] not in ["U25", "BTTS_N", "X"]]

            if bons_paris:
                top = sorted(bons_paris, key=lambda x: x["prob"], reverse=True)[0]
                ticket.append({"match": f"{m['dom']} vs {m['ext']}", "ligue": m["ligue"], "pari": top["pari"], "cote": top["cote"], "prob": top["prob"]})
                vus.add(m["id"])
                if len(ticket) == max_matchs: break
        return ticket

    t_secu = fabriquer_ticket(1.40, 1.65, 60.0, 3)
    t_value = fabriquer_ticket(1.65, 2.10, 48.0, 3)
    t_grosse = fabriquer_ticket(1.80, 2.60, 40.0, 4)

    onglet_t1, onglet_t2, onglet_t3 = st.tabs(["🛡️ Ticket Solide (@3+)", "⚖️ Ticket Value (@6+)", "💣 Grosse Cote (@15+)"])

    with onglet_t1:
        if t_secu:
            st.metric("Cote Totale", f"@{np.prod([x['cote'] for x in t_secu]):.2f}")
            for item in t_secu: st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f})")
        else: st.info("Matchs insuffisants dans cette fourchette.")
    with onglet_t2:
        if t_value:
            st.metric("Cote Totale", f"@{np.prod([x['cote'] for x in t_value]):.2f}")
            for item in t_value: st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f})")
        else: st.info("Matchs insuffisants dans cette fourchette.")
    with onglet_t3:
        if t_grosse:
            st.metric("Cote Totale", f"@{np.prod([x['cote'] for x in t_grosse]):.2f}")
            for item in t_grosse: st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f})")
        else: st.info("Matchs insuffisants dans cette fourchette.")

    st.markdown("---")
    st.subheader("💬 Chat Tipster Gemini (17+ matchs)")
    for msg in st.session_state.messages_chat:
        with st.chat_message(msg["role"]): st.write(msg["content"])
    
    user_prompt = st.chat_input("Ex: 'Fais-moi un combiné de 17 matchs avec une cote de 1000'...")
    if user_prompt:
        st.session_state.messages_chat.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"): st.write(user_prompt)

        ctx = "\n".join([f"{m['dom']}-{m['ext']}|1:@{m['res']['cotes_1N2'][0]:.2f}|2:@{m['res']['cotes_1N2'][2]:.2f}|O25:@{m['res']['cote_o25']:.2f}|BTTS:@{m['res']['cote_btts']:.2f}|BCM:@{m['res']['cote_bcm']:.2f}" for m in matchs_semaine[:35]])
        prompt_systeme = f"Expert paris sportifs. Calendrier:\n{ctx}\nDemande: {user_prompt}\nRédige le ticket exact demandé avec équipes, paris variés et cote totale multipliée."
        with st.spinner("Gemini compose votre combiné..."): reponse_ia = appeler_gemini_ai(prompt_systeme)
        st.session_state.messages_chat.append({"role": "assistant", "content": reponse_ia})
        with st.chat_message("assistant"): st.write(reponse_ia)

# ==========================================
# 7. ONGLET 5 : SCANNER INSTANTANÉ (ANTI-COTES MORTES)
# ==========================================
with onglets[4]:
    
    # --- NOUVEAU : EXTRACTION DES 6 MEILLEURS SCORES EXACTS ---
    tous_scores_exacts = []
    lignes_tableau = []
    
    for m in matchs_semaine:
        r = m["res"]
        top, score_aligne = selectionner_meilleur_pari_value(r, m["paris"])
        score_1, p_s1 = score_aligne
        
        # Enregistrement pour le Top 6 (on garde la date, les équipes, le score et la proba)
        tous_scores_exacts.append({
            "match": f"{m['dom']} vs {m['ext']}",
            "date": m["date_str"],
            "score": score_1,
            "prob": p_s1 * 100,
            "pari": top['pari']
        })

        t = top["type"]
        if t in ["1", "2"]: ia_txt = f"Avantage tactique net (xG {r['lambda_dom']:.1f}-{r['lambda_ext']:.1f})."
        elif t == "X": ia_txt = f"Forte neutralisation attendue (xG proches)."
        elif t in ["O25", "BCM", "O15_MT1", "O15_MT2"]: ia_txt = f"Configuration très ouverte : {r['lambda_dom']+r['lambda_ext']:.2f} buts attendus."
        elif t == "U25": ia_txt = f"Match fermé et verrouillé."
        elif t == "BTTS_O": ia_txt = f"Deux attaques prolifiques ({r['btts']:.0f}% BTTS)."
        elif t == "BTTS_N": ia_txt = f"Déséquilibre offensif ou match très défensif."
        else: ia_txt = f"Pari de couverture avec probabilité de {top['prob']:.1f}%."

        lignes_tableau.append({
            "Date": m["date_str"], "Ligue": m["ligue"], "Match": f"{m['dom']} vs {m['ext']}",
            "Pari Recommandé": f"{top['pari']} ({top['prob']:.1f}% | @{top['cote']:.2f})",
            "Score Probable": f"{score_1} ({p_s1*100:.0f}%)",
            "Analyse Tactique": ia_txt
        })
        
    # --- AFFICHAGE DU TOP 6 ---
st.subheader("🎯 Le Top 6 des Scores Exacts (7 Prochains Jours)")
    top_6 = sorted(tous_scores_exacts, key=lambda x: x["prob"], reverse=True)[:6]
    
    if top_6:
        cols = st.columns(3)
        for i, sc in enumerate(top_6):
            with cols[i % 3]:
                st.info(f"**{sc['match']}**\n\n🎯 **Score Exact : {sc['score']}**\n\n*Fiabilité : {sc['prob']:.1f}%*")
    
    st.markdown("---")

    st.subheader("📅 Scanner de Valeur (Uniquement Cotes $\ge$ @1.40)")
    if lignes_tableau:
        df_scan = pd.DataFrame(lignes_tableau)
        st.success(f"{len(df_scan)} matchs analysés. Diversité de scores activée.")
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
