
# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
import re
from pathlib import Path

st.set_page_config(page_title="ETA – Benchmark qualité (GTFS-rt)", layout="wide")

st.title("🚍 ETA – Qualité & Fiabilité (Benchmark)")
st.caption("Chargez votre fichier (CSV/XLSX) avec les colonnes: agency, route, bucket, source, Early, Late, Accurate, Predictions")

# ------------------------------
# Sidebar – paramètres & filtres
# ------------------------------
with st.sidebar:
    st.header("Paramètres d'analyse")
    st.markdown("**1) Fichier & filtres de base**")

uploaded = st.file_uploader("Déposez un fichier .csv ou .xlsx", type=["csv","xlsx"])   
use_default = st.checkbox("Utiliser le fichier local par défaut: accuracy_general_literal.csv (s'il existe)")

EXPECTED_COLS = {"agency","route","bucket","source","Early","Late","Accurate","Predictions"}
BUCKETS_DEFAULT = ["0 - 3 min","3 - 6 min","6 - 10 min","10 - 15 min"]

# Palette demandée: Précis=vert, En avance=rouge, En retard=orange
COLOR_MAP = {
    'Précis': '#d62728',       # vert
    'En avance': '#ff7f0e',    # rouge
    'En retard': '#2ca02c'     # orange
}

# ---------------- Helpers ----------------

def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise colonnes/types, buckets et calcule un groupe de ligne."""
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    missing = EXPECTED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes: {', '.join(sorted(missing))}")
    # Casts
    for col in ["Early","Late","Accurate","Predictions"]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['route']  = df['route'].astype(str)
    df['bucket'] = df['bucket'].astype(str).str.strip()
    df['agency'] = df['agency'].astype(str).str.strip()
    df['source'] = df['source'].astype(str).str.strip()
    # Normalisation bucket (espaces/tirets)
    def norm_bucket(val: str) -> str:
        s = (val or '').strip()
        s = s.replace('–','-').replace('—','-')
        s = re.sub(r"\s*-\s*", " - ", s)
        return s
    df['bucket'] = df['bucket'].map(norm_bucket)
    # Groupe de lignes (priorité: Aéroport > Haute fréquence > Nuit(300) > Express(400) > Navette(700/800) > Autre)
    HF_SET = {"18","24","51","67","105","121","141","165","439"}
    def line_group(route_str: str) -> str:
        try:
            f = float(route_str)
            n = int(f)
        except Exception:
            return 'Autre'
        if n == 747:
            return 'Aéroport'
        if route_str in HF_SET or str(n) in HF_SET:
            return 'Haute fréquence'
        if 300 <= n < 400:
            return 'Nuit'
        if 400 <= n < 500:
            return 'Express'
        if 700 <= n < 900:
            return 'Navette'
        return 'Autre'
    df['group'] = df['route'].apply(line_group)
    # Nettoyage lignes invalides
    df = df.dropna(subset=["Early","Late","Accurate","Predictions"]) 
    return df

@st.cache_data(show_spinner=False)
def _load_df(file):
    """Accepte un fichier uploadé (Streamlit) OU un chemin local (Path/str)."""
    if file is None:
        return None
    if isinstance(file, (str, Path)):
        p = Path(file)
        if p.suffix.lower() == '.csv':
            df = pd.read_csv(p)
        else:
            df = pd.read_excel(p)
        return _standardize_columns(df)
    name = getattr(file, 'name', '').lower()
    if name.endswith('.csv'):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)
    return _standardize_columns(df)


def weighted_avg(values, weights):
    v = np.asarray(values)
    w = np.asarray(weights)
    if np.nansum(w) == 0:
        return np.nan
    return np.average(v, weights=w)

# ---------- Chargement des données ----------
if uploaded is not None:
    try:
        df = _load_df(uploaded)
    except Exception as e:
        st.error(str(e))
        st.stop()
elif use_default:
    demo_path = Path('accuracy_general_literal.csv')
    if demo_path.exists():
        try:
            df = _load_df(demo_path)
            st.info("Fichier par défaut chargé: accuracy_general_literal.csv")
        except Exception as e:
            st.error(str(e))
            st.stop()
    else:
        st.warning("Le fichier par défaut n'est pas présent dans le répertoire de l'app. Veuillez en téléverser un.")
        st.stop()
else:
    st.info("Chargez un fichier pour lancer l'analyse, ou cochez l'option fichier par défaut si disponible.")
    st.stop()

# --------- Filtres dynamiques ---------
with st.sidebar:
    st.markdown("**2) Filtres**")
    agencies = sorted(df['agency'].dropna().unique().tolist())
    sources = sorted(df['source'].dropna().unique().tolist())
    groups = ['Haute fréquence','Aéroport','Nuit','Express','Navette','Autre']
    groups_present = [g for g in groups if g in df['group'].unique()]
    buckets_all = [b for b in BUCKETS_DEFAULT if b in df['bucket'].unique()]

    agency_sel = st.multiselect("Agence(s)", agencies, default=agencies)
    source_sel = st.multiselect("Source(s)", sources, default=sources)
    group_sel  = st.multiselect("Groupe(s) de lignes", groups_present, default=groups_present,
                                help="Règles: 300=Nuit, 400=Express, 700-800=Navette, 747=Aéroport, HF={18,24,51,67,105,121,141,165,439}")
    buckets_sel = st.multiselect("Horizon(s) inclus pour les KPIs", buckets_all, default=buckets_all)

    st.markdown("**Filtre lignes**")
    route_filter = st.text_input("Contient / Regex (ex: ^[1-9]\\d*$ pour numériques)", value="")
    cand_routes = df['route'].unique().tolist()
    if route_filter.strip():
        try:
            pattern = re.compile(route_filter.strip())
            cand_routes = [r for r in cand_routes if pattern.search(r)]
        except Exception:
            cand_routes = [r for r in cand_routes if route_filter.strip().lower() in r.lower()]
    cand_routes = sorted(cand_routes, key=lambda x: (len(x), x))
    route_sel = st.multiselect("Ligne(s)", cand_routes, default=cand_routes)

    st.markdown("**Seuils & affichage**")
    thresh = st.number_input("Seuil min. de volume Top/Bottom (Predictions)", min_value=0, value=300_000, step=50_000, format="%d")
    show_n = st.slider("Nombre de lignes à afficher (Top & Bottom)", min_value=5, max_value=20, value=10, step=1)

# Appliquer filtres
filt = df[
    df['agency'].isin(agency_sel) &
    df['source'].isin(source_sel) &
    df['group'].isin(group_sel) &
    df['route'].isin(route_sel)
].copy()

# ---------------------
# Calculs des indicateurs
# ---------------------
rows = filt[filt['bucket'].isin(buckets_sel)].copy()

if rows.empty:
    st.warning("Aucune donnée après application des filtres.")
    st.stop()

overall = {
    'Early_%': weighted_avg(rows['Early'], rows['Predictions']),
    'Late_%': weighted_avg(rows['Late'], rows['Predictions']),
    'Accurate_%': weighted_avg(rows['Accurate'], rows['Predictions']),
    'Predictions': rows['Predictions'].sum()
}

by_bucket = (
    rows.groupby('bucket').apply(lambda g: pd.Series({
        'Early_%': weighted_avg(g['Early'], g['Predictions']),
        'Late_%': weighted_avg(g['Late'], g['Predictions']),
        'Accurate_%': weighted_avg(g['Accurate'], g['Predictions']),
        'Predictions': g['Predictions'].sum()
    })).reset_index()
)

# Overall par ligne (Equally Weighted Predictions si fourni)
overall_rows = filt[filt['bucket']=="Overall average: Equally Weighted Predictions"][['agency','route','source','group','Early','Late','Accurate','Predictions']].copy()
if overall_rows.empty:
    agg = rows.groupby(['agency','route','source','group']).apply(lambda g: pd.Series({
        'Early': weighted_avg(g['Early'], g['Predictions']),
        'Late': weighted_avg(g['Late'], g['Predictions']),
        'Accurate': weighted_avg(g['Accurate'], g['Predictions']),
        'Predictions': g['Predictions'].sum()
    })).reset_index()
    overall_rows = agg

# -----------------
# Tuiles KPI globales
# -----------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("% Précises (pondéré)", f"{overall['Accurate_%']:.1f}%")
c2.metric("% En avance (pondéré)", f"{overall['Early_%']:.1f}%")
c3.metric("% En retard (pondéré)", f"{overall['Late_%']:.1f}%")
c4.metric("Volume de prédictions", f"{overall['Predictions']:,}")

st.divider()

# -----------------
# Visuel par bucket
# -----------------
st.subheader("Par horizon/bucket")
order = [b for b in BUCKETS_DEFAULT if b in by_bucket['bucket'].unique()]
by_bucket['bucket'] = pd.Categorical(by_bucket['bucket'], categories=order, ordered=True)
by_bucket = by_bucket.sort_values('bucket')

chart_df = by_bucket.melt(id_vars=['bucket','Predictions'], value_vars=['Accurate_%','Early_%','Late_%'], var_name='Type', value_name='Pourcentage')
chart_df['Type'] = chart_df['Type'].map({'Accurate_%':'Précis','Early_%':'En avance','Late_%':'En retard'})

line = (
    alt.Chart(chart_df)
    .mark_line(point=True)
    .encode(
        x=alt.X('bucket:N', title='Horizon avant arrivée'),
        y=alt.Y('Pourcentage:Q', title='Pourcentage pondéré', scale=alt.Scale(domain=[0,100])),
        color=alt.Color('Type:N', title='Catégorie', scale=alt.Scale(range=[COLOR_MAP['Précis'], COLOR_MAP['En avance'], COLOR_MAP['En retard']])),
        tooltip=['bucket','Type',alt.Tooltip('Pourcentage:Q', format='.1f'),'Predictions']
    ).properties(height=300)
)
st.altair_chart(line, use_container_width=True)

st.dataframe(by_bucket.rename(columns={'bucket':'Bucket','Accurate_%':'% Précis','Early_%':'% En avance','Late_%':'% En retard','Predictions':'Prédictions'}), use_container_width=True)

st.caption("Couleurs: **Précis=vert**, **En avance=rouge**, **En retard=orange**.")

st.divider()

# --------------------------------------
# Visuel comparatif Top/Bottom (empilé)
# --------------------------------------
st.subheader("Comparatif Top / Bottom (≥ seuil de volume)")

def clean_route(x):
    try:
        f = float(x)
        if f.is_integer():
            return str(int(f))
        return str(x)
    except:
        return str(x)

hi = overall_rows[overall_rows['Predictions']>=thresh].copy()
hi['route_display'] = hi['route'].apply(clean_route)

topN = hi.sort_values('Accurate', ascending=False).head(show_n)
botN = hi.sort_values('Accurate', ascending=True).head(show_n)

st.caption(f"{hi['route'].nunique()} lignes passent le seuil (sur {overall_rows['route'].nunique()} lignes filtrées). Seuil = {thresh:,} prédictions.")

# Stacked bar (composition ETA)

def stacked_bar(df_in, title):
    if df_in.empty:
        return alt.Chart(pd.DataFrame({'x':[0]})).mark_text(text='Aucune ligne au-dessus du seuil').properties(title=title)
    d = df_in.copy()
    d = d.rename(columns={'route_display':'Ligne','Accurate':'Précis','Early':'En avance','Late':'En retard'})
    d = d.melt(id_vars=['Ligne','Predictions'], value_vars=['Précis','En avance','En retard'], var_name='Type', value_name='Pourcentage')
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y('Ligne:N', sort='-x', title='Ligne'),
            x=alt.X('Pourcentage:Q', title='Pourcentage (%)', scale=alt.Scale(domain=[0,100])),
            color=alt.Color('Type:N', scale=alt.Scale(range=[COLOR_MAP['Précis'], COLOR_MAP['En avance'], COLOR_MAP['En retard']])),
            tooltip=['Ligne','Type',alt.Tooltip('Pourcentage:Q', format='.1f'),'Predictions']
        )
        .properties(title=title, height=300)
    )
    return chart

c5, c6 = st.columns(2)
with c5:
    st.altair_chart(stacked_bar(topN, 'Top lignes – composition ETA (%)'), use_container_width=True)
with c6:
    st.altair_chart(stacked_bar(botN, 'Bottom lignes – composition ETA (%)'), use_container_width=True)

st.divider()

# -----------------------------------------------------
# Scatter: % Précis vs % En avance (taille = volume)
# -----------------------------------------------------
st.subheader("Dispersion des lignes (pondéré)")
scatter_df = hi.copy()
scatter_df = scatter_df.rename(columns={'route_display':'Ligne'})
bubble = (
    alt.Chart(scatter_df)
    .mark_circle(opacity=0.7)
    .encode(
        x=alt.X('Early:Q', title='% En avance', scale=alt.Scale(domain=[0, max(5, float(scatter_df['Early'].max())+5)])),
        y=alt.Y('Accurate:Q', title='% Précis', scale=alt.Scale(domain=[0, 100])),
        size=alt.Size('Predictions:Q', title='Volume prédictions', scale=alt.Scale(range=[20, 800])),
        color=alt.Color('group:N', title='Groupe', legend=alt.Legend(orient='right')),
        tooltip=['Ligne','group','agency','source',alt.Tooltip('Accurate:Q', format='.1f'),alt.Tooltip('Early:Q', format='.1f'),alt.Tooltip('Late:Q', format='.1f'),'Predictions']
    )
    .properties(height=380)
    .interactive()
)
st.altair_chart(bubble, use_container_width=True)

st.divider()

# -----------------------------------------------------
# Profil d'une ligne sélectionnée – par bucket
# -----------------------------------------------------
st.subheader("Profil détaillé d'une ligne par horizon")
selectable_routes = sorted(rows['route'].unique().tolist(), key=lambda x: (len(x), x))
route_focus = st.selectbox("Choisir une ligne", options=selectable_routes)

prof = rows[rows['route']==route_focus].copy()
if not prof.empty:
    prof['bucket'] = pd.Categorical(prof['bucket'], categories=order, ordered=True)
    prof = prof.sort_values('bucket')
    prof_long = prof.melt(id_vars=['bucket','Predictions'], value_vars=['Accurate','Early','Late'], var_name='Type', value_name='Pourcentage')
    prof_long['Type'] = prof_long['Type'].map({'Accurate':'Précis','Early':'En avance','Late':'En retard'})
    bar_prof = (
        alt.Chart(prof_long)
        .mark_bar()
        .encode(
            x=alt.X('bucket:N', title='Horizon'),
            y=alt.Y('Pourcentage:Q', title='Pourcentage (%)', scale=alt.Scale(domain=[0,100])),
            color=alt.Color('Type:N', scale=alt.Scale(range=[COLOR_MAP['Précis'], COLOR_MAP['En avance'], COLOR_MAP['En retard']])),
            tooltip=['bucket','Type',alt.Tooltip('Pourcentage:Q', format='.1f'),'Predictions']
        )
        .properties(title=f"Ligne {route_focus} – composition par horizon", height=320)
    )
    st.altair_chart(bar_prof, use_container_width=True)
else:
    st.info("Aucune donnée par bucket pour cette ligne avec les filtres actuels.")

st.divider()

# -------------------------------
# Téléchargements des jeux de données
# -------------------------------
st.subheader("Données détaillées – téléchargement")

overall_dl = overall_rows[['agency','route','source','group','Early','Late','Accurate','Predictions']].copy().sort_values('route')
st.download_button(
    label="Télécharger – Overall par ligne (CSV)",
    data=overall_dl.to_csv(index=False).encode('utf-8'),
    file_name='overall_by_route.csv',
    mime='text/csv'
)

buckets_rows = rows.copy()
st.download_button(
    label="Télécharger – Par horizon / par ligne (CSV)",
    data=buckets_rows.to_csv(index=False).encode('utf-8'),
    file_name='by_horizon_by_route.csv',
    mime='text/csv'
)
