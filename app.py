import streamlit as st
import pandas as pd
import numpy as np
import requests
from scipy.stats import poisson
import difflib
from datetime import datetime, timedelta, timezone

# ==========================================
# CONFIGURATION & API
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
# 1. DICTIONNAIRE EXACT
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
    if not nom_api or len(equipes_autorisees) == 0:
        return None
    if nom_api in CORRESPONDANCES_NOMS:
        c = CORRESPONDANCES_NOMS[nom_api]
        if c in equipes_autorisees: return c

    mots = ["FC", "CF", "AS", "SSC", "AC", "SC", "SV", "RC", "RCD", "UD", "CD", "AFC", "1909", "1904", "1899", "1846", "1848", "1910", "1913", "1907", "04", "05"]
    nom_net = nom_api
    for m in mots:
        nom_net = nom_net.replace(f" {m} ", " ").replace(f" {m}", "").replace(f"{m} ", "")
    nom_net = nom_net.strip()

    for eq in equipes_autorisees:
        if eq.lower() == nom_net.lower() or eq.lower() == nom_api.lower(): return eq
    for eq in equipes_autorisees:
        if len(eq) >= 4 and (eq.lower() in nom_net.lower() or nom_net.lower() in eq.lower()): return eq

    p = difflib.get_close_matches(nom_net, equipes_autorisees, n=1, cutoff=0.58)
    return p[0] if p else None

# ==========================================
# 2. APPEL GEMINI AI AVEC HISTORIQUE CHAT
# ==========================================
def appeler_gemini_ai(prompt_texte):
    headers_ia = {"Content-Type": "application/json", "X-goog-api-key": GEMINI_API_KEY}
    payload = {"contents": [{"parts": [{"text": prompt_texte}]}]}
    try:
        r = requests.post(GEMINI_URL, headers=headers_ia, json=payload, timeout=10)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        pass
    return None

# ==========================================
# 3. STATISTIQUES & MODÈLE POISSON + DIXON-COLES
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

    df_ligue = df_total[df_total['Championnat'] == matchs['Championnat'].iloc[0]]
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
    r_mt_d, r_mt_e = 0.44, 0.44
    if 'Buts_Dom_MT' in df_ligue.columns and 'Buts_Ext_MT' in df_ligue.columns:
        tot_d, tot_e = m_dom['Buts_Dom'].sum(), m_ext['Buts_Ext'].sum()
        if tot_d > 5: r_mt_d = np.clip(m_dom['Buts_Dom_MT'].sum() / tot_d, 0.30, 0.60)
        if tot_e > 5: r_mt_e = np.clip(m_ext['Buts_Ext_MT'].sum() / tot_e, 0.30, 0.60)

    return att_d, def_d, att_e, def_e, moy_dom, moy_ext, r_mt_d, r_mt_e, indice_forme

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

    p_v1, p_x, p_v2 = np.sum(np.tril(mat_ft, -1))*100, np.sum(np.diag(mat_ft))*100, np.sum(np.triu(mat_ft, 1))*100
    p_1x, p_x2 = p_v1 + p_x, p_v2 + p_x

    over_15 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 1.5) * 100
    over_25 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 2.5) * 100
    over_35 = sum(p for (i, j), p in np.ndenumerate(mat_ft) if (i + j) > 3.5) * 100
    under_25, under_35 = 100.0 - over_25, 100.0 - over_35
    btts_oui = np.sum(mat_ft[1:, 1:]) * 100
    top_scores = sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)[:5]

    mat_mt = np.zeros((5, 5))
    l_d_mt, l_e_mt = r_mt_d * lambda_dom, r_mt_e * lambda_ext
    for i in range(5):
        for j in range(5): mat_mt[i, j] = poisson.pmf(i, l_d_mt) * poisson.pmf(j, l_e_mt)
    mat_mt = mat_mt / np.sum(mat_mt)

    over_05_mt = (1.0 - mat_mt[0, 0]) * 100
    v1_mt, x_mt, v2_mt = np.sum(np.tril(mat_mt, -1))*100, np.sum(np.diag(mat_mt))*100, np.sum(np.triu(mat_mt, 1))*100

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

def analyser_paris_match(dom, ext, res):
    """Génère tous les marchés disponibles pour un match avec cotes et probabilités."""
    return [
        {"pari": f"Victoire {dom}", "prob": res["1N2"][0], "cote": res["cotes_1N2"][0], "type": "1"},
        {"pari": f"Victoire {ext}", "prob": res["1N2"][2], "cote": res["cotes_1N2"][2], "type": "2"},
        {"pari": "Over 2.5 Buts", "prob": res["over_25"], "cote": res["cote_o25"], "type": "O25"},
        {"pari": "Les 2 Marquent (BTTS)", "prob": res["btts"], "cote": res["cote_btts"], "type": "BTTS"},
        {"pari": f"1X ({dom} ou Nul)", "prob": res["1X_X2"][0], "cote": res["cotes_DC"][0], "type": "1X"},
        {"pari": f"X2 ({ext} ou Nul)", "prob": res["1X_X2"][1], "cote": res["cotes_DC"][1], "type": "X2"},
        {"pari": "Over 1.5 Buts", "prob": res["over_15"], "cote": res["cote_o15"], "type": "O15"},
        {"pari": "Under 3.5 Buts", "prob": res["under_35"], "cote": res["cote_u35"], "type": "U35"}
    ]

# ==========================================
# 4. COLLECTE STRICTE 7 JOURS
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
            try: dt_match = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            except Exception: continue
            if not (maintenant <= dt_match <= limite_7j): continue

            dom_csv = trouver_nom_csv_strict(m["homeTeam"]["name"], equipes_autorisees)
            ext_csv = trouver_nom_csv_strict(m["awayTeam"]["name"], equipes_autorisees)
            if not dom_csv or not ext_csv or dom_csv == ext_csv: continue

            res = predire_rencontre(dom_csv, ext_csv, est_europe=est_europe)
            tous_paris = analyser_paris_match(dom_csv, ext_csv, res)

            liste_propre.append({
                "id": m["id"], "dt": dt_match,
                "date_str": dt_match.strftime("%d/%m à %H:%M"),
                "jour_date": dt_match.date(), "heure_str": dt_match.strftime("%H:%M"),
                "ligue": nom_compet, "dom": dom_csv, "ext": ext_csv,
                "est_europe": est_europe, "res": res, "paris": tous_paris
            })
    return sorted(liste_propre, key=lambda x: x["dt"])

# ==========================================
# 5. INTERFACE MOBILE
# ==========================================
st.markdown("<h2 style='text-align: center;'>⚽ DIRECT FOOT PRO AI</h2>", unsafe_allow_html=True)

onglets = st.tabs(["Aujourd'hui", "Demain", "Jours Suivants", "🎟️ Tickets & Chat AI", "📅 Scanner 7J"])
matchs_semaine = collecter_matchs_7_jours()
date_auj = datetime.now(timezone.utc).date()
date_dem = date_auj + timedelta(days=1)

if "match_actif" not in st.session_state:
    st.session_state.match_actif = None
if "messages_chat" not in st.session_state:
    st.session_state.messages_chat = []

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

# Fiche détaillée
if st.session_state.match_actif:
    m_sel = st.session_state.match_actif
    res = predire_rencontre(m_sel["dom"], m_sel["ext"], est_europe=m_sel["est_europe"])
    st.markdown("---")
    st.subheader(f"🏟️ {m_sel['dom']} vs {m_sel['ext']} ({m_sel['ligue']})")
    st.caption(f"📅 Coup d'envoi : {m_sel['date_str']} UTC")

    c1, c2, c3 = st.columns(3)
    c1.metric(f"1 ({m_sel['dom']})", f"{res['1N2'][0]:.1f}%", f"@{res['cotes_1N2'][0]:.2f}")
    c2.metric("Nul (X)", f"{res['1N2'][1]:.1f}%", f"@{res['cotes_1N2'][1]:.2f}")
    c3.metric(f"2 ({m_sel['ext']})", f"{res['1N2'][2]:.1f}%", f"@{res['cotes_1N2'][2]:.2f}")

    dec_btts = "OUI" if res['btts'] >= 50 else "NON"
    st.write(f"• **BTTS** : `{dec_btts}` ({res['btts']:.1f}% @{res['cote_btts']:.2f}) | **Over 2.5** : `{res['over_25']:.1f}%` (@{res['cote_o25']:.2f})")
    st.write(f"• **Score le plus probable** : `{res['top_scores'][0][0]}` ({res['top_scores'][0][1]*100:.1f}%)")

# ==========================================
# 6. ONGLET 4 : MULTI-TICKETS & CHAT INTERACTIF GEMINI
# ==========================================
with onglets[3]:
    st.subheader("🎟️ Générateur Multi-Tickets (Cotes @2, @5 - @7, @10+)")

    # Pool de matchs : Aujourd'hui ou bascule Demain
    pool = [m for m in matchs_semaine if m["jour_date"] == date_auj]
    tag_date = "Aujourd'hui"
    if len(pool) < 4:
        pool = [m for m in matchs_semaine if m["jour_date"] in [date_auj, date_dem]]
        tag_date = "Aujourd'hui & Demain"
    if len(pool) < 4:
        pool = matchs_semaine[:10]
        tag_date = "Prochains jours"

    st.caption(f"Sélection basée sur les matchs de : **{tag_date}**")

    # Fonction de composition de ticket
    def fabriquer_ticket(min_cote, max_cote, min_prob, max_matchs):
        ticket = []
        matchs_utilises = set()
        for m in pool:
            if m["id"] in matchs_utilises: continue
            bons_paris = [p for p in m["paris"] if min_cote <= p["cote"] <= max_cote and p["prob"] >= min_prob]
            if bons_paris:
                top_p = sorted(bons_paris, key=lambda x: x["prob"], reverse=True)[0]
                ticket.append({
                    "match": f"{m['dom']} vs {m['ext']}", "ligue": m["ligue"],
                    "pari": top_p["pari"], "cote": top_p["cote"], "prob": top_p["prob"]
                })
                matchs_utilises.add(m["id"])
                if len(ticket) == max_matchs: break
        return ticket

    t_secu = fabriquer_ticket(1.22, 1.45, 70.0, 3)
    t_value = fabriquer_ticket(1.50, 1.95, 52.0, 3)
    t_grosse = fabriquer_ticket(1.65, 2.35, 45.0, 4)

    onglet_t1, onglet_t2, onglet_t3 = st.tabs(["🛡️ Ticket Sécu (@2 - @3)", "⚖️ Ticket Value (@5 - @7)", "💣 Grosse Cote (@10+)"])

    with onglet_t1:
        if t_secu:
            cote_tot = np.prod([x["cote"] for x in t_secu])
            st.metric("Cote Totale", f"@{cote_tot:.2f}")
            for item in t_secu:
                st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f} — {item['prob']:.1f}%)")
        else: st.info("Pas assez de matchs à très haute sécurité aujourd'hui.")

    with onglet_t2:
        if t_value:
            cote_tot = np.prod([x["cote"] for x in t_value])
            st.metric("Cote Totale", f"@{cote_tot:.2f}")
            for item in t_value:
                st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f} — {item['prob']:.1f}%)")
        else: st.info("Aucune sélection dans cette fourchette.")

    with onglet_t3:
        if t_grosse:
            cote_tot = np.prod([x["cote"] for x in t_grosse])
            st.metric("Cote Totale", f"@{cote_tot:.2f}")
            for item in t_grosse:
                st.write(f"• **{item['match']}** : `{item['pari']}` (@{item['cote']:.2f} — {item['prob']:.1f}%)")
        else: st.info("Pas assez de matchs pour une cote @10+.")

    st.markdown("---")
    st.subheader("💬 Discutez avec Gemini pour modifier ou créer vos tickets")
    st.caption("Demandez-lui : *'Fais-moi un ticket cote 8 avec des Over 2.5'*, *'Remplace un match'*, etc.")

    # Affichage de l'historique du chat
    for msg in st.session_state.messages_chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Zone de saisie du message
    user_prompt = st.chat_input("Écrivez votre consigne à Gemini...")
    if user_prompt:
        st.session_state.messages_chat.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.write(user_prompt)

        # Préparation du contexte des matchs disponibles pour Gemini
        contexte_matchs = "\n".join([
            f"- {m['dom']} vs {m['ext']} ({m['ligue']}) : xG={m['res']['lambda_dom']:.2f}-{m['res']['lambda_ext']:.2f}, "
            f"1N2=({m['res']['1N2'][0]:.0f}%/{m['res']['1N2'][1]:.0f}%/{m['res']['1N2'][2]:.0f}%), "
            f"Over2.5={m['res']['over_25']:.0f}% (@{m['res']['cote_o25']:.2f}), BTTS={m['res']['btts']:.0f}% (@{m['res']['cote_btts']:.2f})"
            for m in pool[:12]
        ])

        prompt_systeme = (
            f"Tu es un tipster professionnel en direct avec l'utilisateur. Tu as accès aux matchs récents du calendrier et aux cotes mathématiques :\n"
            f"{contexte_matchs}\n\n"
            f"L'utilisateur te demande : '{user_prompt}'. "
            f"Réponds en français de façon claire, percutante et propose un ticket sur mesure (avec les matchs, les paris précis, les cotes et la cote totale combinée)."
        )

        with st.spinner("Gemini compose votre ticket..."):
            reponse_ia = appeler_gemini_ai(prompt_systeme) or "Désolé, je n'ai pas pu joindre le serveur. Réessayez."

        st.session_state.messages_chat.append({"role": "assistant", "content": reponse_ia})
        with st.chat_message("assistant"):
            st.write(reponse_ia)

# ==========================================
# 7. ONGLET 5 : SCANNER INSTANTANÉ (ULTRA-RAPIDE)
# ==========================================
with onglets[4]:
    st.subheader("📅 Scanner Rapide des 7 Prochains Jours")

    lignes_tableau = []
    for m in matchs_semaine:
        r = m["res"]
        # Meilleur pari instantané
        p_candidats = [
            ("Over 2.5", r["over_25"], r["cote_o25"]),
            (f"Victoire {m['dom']}", r["1N2"][0], r["cotes_1N2"][0]),
            (f"Victoire {m['ext']}", r["1N2"][2], r["cotes_1N2"][2]),
            ("BTTS (Oui)", r["btts"], r["cote_btts"]),
            ("Over 1.5", r["over_15"], r["cote_o15"])
        ]
        top = sorted(p_candidats, key=lambda x: x[1], reverse=True)[0]
        score_1, p_s1 = r["top_scores"][0]

        # Analyse instantanée en Python (0 milliseconde)
        if top[0].startswith("Victoire"):
            ia_txt = f"Avantage xG ({r['lambda_dom']:.1f}-{r['lambda_ext']:.1f}) et dynamique récente à {r['forme_dom']:+.0f}%."
        elif "Over" in top[0]:
            ia_txt = f"Match ouvert : {r['lambda_dom']+r['lambda_ext']:.2f} buts attendus cumulés."
        else:
            ia_txt = f"Deux attaques prolifiques ({r['btts']:.0f}% de chances BTTS)."

        lignes_tableau.append({
            "Date": m["date_str"],
            "Ligue": m["ligue"],
            "Match": f"{m['dom']} vs {m['ext']}",
            "Pari Recommandé": f"{top[0]} ({top[1]:.1f}% | @{top[2]:.2f})",
            "Score Probable": f"{score_1} ({p_s1*100:.0f}%)",
            "Analyse Tactique": ia_txt
        })

    if lignes_tableau:
        df_scan = pd.DataFrame(lignes_tableau)
        st.success(f"{len(df_scan)} matchs analysés instantanément.")
        st.dataframe(df_scan, use_container_width=True, hide_index=True)
