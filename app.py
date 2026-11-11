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
    # --- Top 5 Europe & LDC ---
    "UEFA Champions League": "CL",
    "Premier League": "PL",
    "La Liga": "PD",
    "Serie A": "SA",
    "Bundesliga": "BL1",
    "Ligue 1": "FL1",
    
    # --- Autres Ligues ---
    "Championship (Ang.)": "ELC",
    "Eredivisie (Pays-Bas)": "DED",
    "Primeira Liga (Portugal)": "PPL",
    "Brasileirão (Brésil)": "BSA",
    "Copa Libertadores": "CLI",

    # --- Équipes Nationales & Tournois ---
    "Coupe du Monde": "WC",
    "Euro (Nations)": "EC",
    "Ligue des Nations UEFA": "UNL",
    "Amicaux Internationaux": "FINT"
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
    if not nom_api:
        return None
    if nom_api in CORRESPONDANCES_NOMS:
        c = CORRESPONDANCES_NOMS[nom_api]
        if len(equipes_autorisees) == 0 or c in equipes_autorisees:
            return c

    mots = ["FC", "CF", "AS", "SSC", "AC", "SC", "SV", "RC", "RCD", "UD", "CD", "AFC", "1909", "1904", "1899", "1846", "1848", "1910", "1913", "1907", "04", "05"]
    nom_net = nom_api
    for m in mots:
        nom_net = nom_net.replace(f" {m} ", " ").replace(f" {m}", "").replace(f"{m} ", "")
    nom_net = nom_net.strip()

    if len(equipes_autorisees) > 0:
        for eq in equipes_autorisees:
            if eq.lower() == nom_net.lower() or eq.lower() == nom_api.lower():
                return eq
        for eq in equipes_autorisees:
            if len(eq) >= 4 and (eq.lower() in nom_net.lower() or nom_net.lower() in eq.lower()):
                return eq
        p = difflib.get_close_matches(nom_net, equipes_autorisees, n=1, cutoff=0.58)
        if p:
            return p[0]

    # Si la ligue ou l'équipe nationale n'est pas dans le CSV, on retourne le nom nettoyé de l'API
    return nom_net if nom_net else nom_api

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
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            url_alt = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent?key={GEMINI_API_KEY}"
            r_alt = requests.post(url_alt, headers=headers, json=payload, timeout=60)
            if r_alt.status_code == 200:
                return r_alt.json()["candidates"][0]["
