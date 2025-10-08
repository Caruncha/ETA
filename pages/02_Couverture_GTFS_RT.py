# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from pathlib import Path
import re

st.set_page_config(page_title="Couverture – GTFS-rt vs planifié (STM)", layout="wide")

st.title("🚌 Couverture – GTFS‑rt vs planifié (STM)")
st.caption(
    "Chargez votre fichier CSV `coverage_with_bands*.csv` (colonnes requises : "
    "startDate, endDate, route, timePeriod, scheduledTripStops, "
    "countTrackedExplained, countOnFullyMissingTrips, countMissingOther, "
    "fractionTrackedExplained, fractionOnFullyMissingTrips, fractionMissingOther)."
)

# -------------------------------------------------------------------
# Sidebar – paramètres & filtres
# -------------------------------------------------------------------
with st.sidebar:
    st.header("Paramètres – Couverture")
    cov_file = st.file_uploader("Déposez le fichier de couverture (.csv)", type=["csv"], key="cov_uploader")
    use_default_cov = st.checkbox(
        "Utiliser un fichier local par défaut (coverage_with_bands.csv) s'il existe",
        value=False
    )

EXPECTED_COLS_COV = {
    "startDate","endDate","route","timePeriod","scheduledTripStops",
    "countTrackedExplained","countOnFullyMissingTrips","countMissingOther",
    "fractionTrackedExplained","fractionOnFullyMissingTrips","fractionMissingOther"
}

TIME_ORDER = ["Rush AM", "Rush PM", "Off-Peak", "All"]

COLOR_MAP = {
    "Couvert": "#ff7f0e",      # vert
    "Entièrement manquant": "#2ca02c",  # rouge
    "Autres manquants": "#d62728"       # orange
}

def _load_coverage(file_or_path):
    if file_or_path is None:
        return None
    if isinstance(file_or_path, (str, Path)):
        df = pd.read_csv(file_or_path)
    else:
        df = pd.read_csv(file_or_path)

    # Standardisation colonnes/types
    df.columns = [c.strip() for c in df.columns]
    missing = EXPECTED_COLS_COV - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans le CSV: {', '.join(sorted(missing))}")

    num_cols = [
        "scheduledTripStops",
        "countTrackedExplained",
        "countOnFullyMissingTrips",
        "countMissingOther",
        "fractionTrackedExplained",
        "fractionOnFullyMissingTrips",
        "fractionMissingOther",
    ]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["route"] = df["route"].astype(str).str.strip()
    df["timePeriod"] = df["timePeriod"].astype(str).str.strip()
    # Normalisation cohérente des périodes
    df["timePeriod"] = df["timePeriod"].replace({
        "Off-Peak": "Off-Peak",
        "Rush AM": "Rush AM",
        "Rush PM": "Rush PM",
        "All": "All"
    })
    # Sécurité : pas de divisions par zéro
    df["scheduledTripStops"] = df["scheduledTripStops"].fillna(0)

    # (Re)calcule les pourcentages à partir des comptes (plus robuste)
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = df["scheduledTripStops"].replace(0, np.nan)
        df["pct_tracked"] = (df["countTrackedExplained"] / denom * 100).fillna(0)
        df["pct_full_missing"] = (df["countOnFullyMissingTrips"] / denom * 100).fillna(0)
        df["pct_other_missing"] = (df["countMissingOther"] / denom * 100).fillna(0)

    return df

# Chargement (upload ou défaut)
df_cov = None
if cov_file is not None:
    try:
        df_cov = _load_coverage(cov_file)
    except Exception as e:
        st.error(f"Erreur de chargement du CSV : {e}")
        st.stop()
elif use_default_cov:
    default_path = Path("coverage_with_bands.csv")
    if default_path.exists():
        try:
            df_cov = _load_coverage(default_path)
            st.info("Fichier par défaut chargé : coverage_with_bands.csv")
        except Exception as e:
            st.error(f"Erreur de chargement du CSV local : {e}")
            st.stop()
    else:
        st.warning("Aucun fichier local `coverage_with_bands.csv` trouvé. Merci d’en téléverser un.")
        st.stop()
else:
    st.info("Déposez un fichier CSV pour lancer l'analyse de couverture.")
    st.stop()

# -------------------------------------------------------------------
# Filtres
# -------------------------------------------------------------------
with st.sidebar:
    st.markdown("**Filtres**")
    periods_present = [p for p in TIME_ORDER if p in sorted(df_cov["timePeriod"].unique())]
    period_sel = st.multiselect("Période(s) temporelle(s)", periods_present, default=periods_present)

    # Filtre regex/contains pour les lignes
    st.markdown("**Filtre lignes**")
    route_filter = st.text_input("Contient / Regex (ex: ^[1-9]\\d*$)", value="")
    candidates = sorted(df_cov["route"].unique(), key=lambda x: (len(x), x))
    if route_filter.strip():
        try:
            pat = re.compile(route_filter.strip())
            candidates = [r for r in candidates if pat.search(r)]
        except Exception:
            s = route_filter.strip().lower()
            candidates = [r for r in candidates if s in r.lower()]

    route_sel = st.multiselect("Ligne(s)", candidates, default=candidates)

    st.markdown("**Seuils & affichage**")
    thresh = st.number_input(
        "Seuil min. volume (arrêts programmés) pour Top/Bottom",
        min_value=0, value=10_000, step=1_000, format="%d"
    )
    show_n = st.slider("Nombre de lignes à afficher (Top & Bottom)", 5, 20, 10, 1)

# Applique filtres
filt = df_cov[
    df_cov["timePeriod"].isin(period_sel) &
    df_cov["route"].isin(route_sel)
].copy()

if filt.empty:
    st.warning("Aucune donnée après application des filtres.")
    st.stop()

# -------------------------------------------------------------------
# KPIs globaux pondérés
# -------------------------------------------------------------------
def _sum_safe(s):  # lisibilité
    return float(pd.to_numeric(s, errors="coerce").fillna(0).sum())

tot_sched = _sum_safe(filt["scheduledTripStops"])
tot_tracked = _sum_safe(filt["countTrackedExplained"])
tot_full_miss = _sum_safe(filt["countOnFullyMissingTrips"])
tot_other_miss = _sum_safe(filt["countMissingOther"])

def pct(x, d):
    return (x / d * 100) if d > 0 else 0.0

kpi_tracked = pct(tot_tracked, tot_sched)
kpi_full_miss = pct(tot_full_miss, tot_sched)
kpi_other_miss = pct(tot_other_miss, tot_sched)

c1, c2, c3, c4 = st.columns(4)
c1.metric("% Couvert (pondéré)", f"{kpi_tracked:.1f}%")
c2.metric("% Entièrement manquant", f"{kpi_full_miss:.1f}%")
c3.metric("% Autres manquants", f"{kpi_other_miss:.1f}%")
c4.metric("Arrêts programmés (total)", f"{int(tot_sched):,}")

st.divider()

# -------------------------------------------------------------------
# Vue par période – empilé
# -------------------------------------------------------------------
st.subheader("Par période temporelle")

by_period = (
    filt.groupby("timePeriod", as_index=False)
        .agg({
            "scheduledTripStops": "sum",
            "countTrackedExplained": "sum",
            "countOnFullyMissingTrips": "sum",
            "countMissingOther": "sum"
        })
)
# Pourcentages pondérés par période
den = by_period["scheduledTripStops"].replace(0, np.nan)
by_period["% Couvert"] = (by_period["countTrackedExplained"] / den * 100).fillna(0)
by_period["% Entièrement manquant"] = (by_period["countOnFullyMissingTrips"] / den * 100).fillna(0)
by_period["% Autres manquants"] = (by_period["countMissingOther"] / den * 100).fillna(0)

# Ordre des périodes
by_period["timePeriod"] = pd.Categorical(by_period["timePeriod"], categories=TIME_ORDER, ordered=True)
by_period = by_period.sort_values("timePeriod")

# Long format pour Altair
chart_df = by_period.melt(
    id_vars=["timePeriod", "scheduledTripStops"],
    value_vars=["% Couvert", "% Entièrement manquant", "% Autres manquants"],
    var_name="Type", value_name="Pourcentage"
)

bar = (
    alt.Chart(chart_df)
    .mark_bar()
    .encode(
        x=alt.X("timePeriod:N", title="Période"),
        y=alt.Y("Pourcentage:Q", title="Pourcentage (%)", scale=alt.Scale(domain=[0, 100])),
        color=alt.Color(
            "Type:N", title="Catégorie",
            scale=alt.Scale(range=[COLOR_MAP["Couvert"], COLOR_MAP["Entièrement manquant"], COLOR_MAP["Autres manquants"]])
        ),
        tooltip=["timePeriod","Type", alt.Tooltip("Pourcentage:Q", format=".1f"), "scheduledTripStops"]
    )
    .properties(height=320)
)
st.altair_chart(bar, use_container_width=True)
st.dataframe(
    by_period.rename(columns={"timePeriod": "Période", "scheduledTripStops": "Arrêts programmés"}),
    use_container_width=True
)

st.divider()

# -------------------------------------------------------------------
# Top/Bottom lignes – composition
# -------------------------------------------------------------------
st.subheader("Comparatif Top / Bottom (≥ seuil de volume)")

# Agrégation par ligne (sur les périodes sélectionnées)
per_route = (
    filt.groupby("route", as_index=False)
        .agg({
            "scheduledTripStops": "sum",
            "countTrackedExplained": "sum",
            "countOnFullyMissingTrips": "sum",
            "countMissingOther": "sum"
        })
)
denr = per_route["scheduledTripStops"].replace(0, np.nan)
per_route["Couvert"] = (per_route["countTrackedExplained"] / denr * 100).fillna(0)
per_route["Entièrement manquant"] = (per_route["countOnFullyMissingTrips"] / denr * 100).fillna(0)
per_route["Autres manquants"] = (per_route["countMissingOther"] / denr * 100).fillna(0)

eligible = per_route[per_route["scheduledTripStops"] >= thresh].copy()
eligible["Ligne"] = eligible["route"]

topN = eligible.sort_values("Couvert", ascending=False).head(show_n)
botN = eligible.sort_values("Couvert", ascending=True).head(show_n)

st.caption(
    f"{eligible['Ligne'].nunique()} lignes passent le seuil "
    f"(sur {per_route['route'].nunique()} lignes filtrées). Seuil = {thresh:,} arrêts programmés."
)

def stacked_routes(df_in, title):
    if df_in.empty:
        return alt.Chart(pd.DataFrame({"x":[0]})).mark_text(text="Aucune ligne au-dessus du seuil").properties(title=title)
    d = df_in[["Ligne","scheduledTripStops","Couvert","Entièrement manquant","Autres manquants"]].copy()
    d = d.melt(
        id_vars=["Ligne","scheduledTripStops"],
        value_vars=["Couvert","Entièrement manquant","Autres manquants"],
        var_name="Type", value_name="Pourcentage"
    )
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            y=alt.Y("Ligne:N", sort="-x", title="Ligne"),
            x=alt.X("Pourcentage:Q", title="Pourcentage (%)", scale=alt.Scale(domain=[0,100])),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(range=[COLOR_MAP["Couvert"], COLOR_MAP["Entièrement manquant"], COLOR_MAP["Autres manquants"]])
            ),
            tooltip=["Ligne","Type", alt.Tooltip("Pourcentage:Q", format=".1f"), "scheduledTripStops"]
        )
        .properties(title=title, height=300)
    )
    return chart

c5, c6 = st.columns(2)
with c5:
    st.altair_chart(stacked_routes(topN, "Top lignes – composition de couverture (%)"), use_container_width=True)
with c6:
    st.altair_chart(stacked_routes(botN, "Bottom lignes – composition de couverture (%)"), use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# Profil détaillé d'une ligne par période
# -------------------------------------------------------------------
st.subheader("Profil détaillé d'une ligne par période")

routes_selectables = sorted(filt["route"].unique(), key=lambda x: (len(x), x))
route_focus = st.selectbox("Choisir une ligne", options=routes_selectables)

prof = filt[filt["route"] == route_focus].copy()
if prof.empty:
    st.info("Aucune donnée pour cette ligne avec les filtres actuels.")
else:
    prof = (
        prof.groupby("timePeriod", as_index=False)
            .agg({
                "scheduledTripStops": "sum",
                "countTrackedExplained": "sum",
                "countOnFullyMissingTrips": "sum",
                "countMissingOther": "sum"
            })
    )
    denp = prof["scheduledTripStops"].replace(0, np.nan)
    prof["Couvert"] = (prof["countTrackedExplained"] / denp * 100).fillna(0)
    prof["Entièrement manquant"] = (prof["countOnFullyMissingTrips"] / denp * 100).fillna(0)
    prof["Autres manquants"] = (prof["countMissingOther"] / denp * 100).fillna(0)

    prof["timePeriod"] = pd.Categorical(prof["timePeriod"], categories=TIME_ORDER, ordered=True)
    prof = prof.sort_values("timePeriod")

    prof_long = prof.melt(
        id_vars=["timePeriod","scheduledTripStops"],
        value_vars=["Couvert","Entièrement manquant","Autres manquants"],
        var_name="Type", value_name="Pourcentage"
    )

    bar_prof = (
        alt.Chart(prof_long)
        .mark_bar()
        .encode(
            x=alt.X("timePeriod:N", title="Période"),
            y=alt.Y("Pourcentage:Q", title="Pourcentage (%)", scale=alt.Scale(domain=[0,100])),
            color=alt.Color(
                "Type:N",
                scale=alt.Scale(range=[COLOR_MAP["Couvert"], COLOR_MAP["Entièrement manquant"], COLOR_MAP["Autres manquants"]])
            ),
            tooltip=["timePeriod","Type", alt.Tooltip("Pourcentage:Q", format=".1f"), "scheduledTripStops"]
        )
        .properties(title=f"Ligne {route_focus} – composition par période", height=320)
    )
    st.altair_chart(bar_prof, use_container_width=True)

st.divider()

# -------------------------------------------------------------------
# Téléchargements
# -------------------------------------------------------------------
st.subheader("Données agrégées – téléchargement")

overall_dl = pd.DataFrame({
    "Arrêts programmés": [int(tot_sched)],
    "% Couvert": [kpi_tracked],
    "% Entièrement manquant": [kpi_full_miss],
    "% Autres manquants": [kpi_other_miss]
})
st.download_button(
    label="Télécharger – KPIs globaux (CSV)",
    data=overall_dl.to_csv(index=False).encode("utf-8"),
    file_name="coverage_kpis_overall.csv",
    mime="text/csv"
)

by_period_dl = by_period.rename(columns={"timePeriod":"Période"})
st.download_button(
    label="Télécharger – Par période (CSV)",
    data=by_period_dl.to_csv(index=False).encode("utf-8"),
    file_name="coverage_by_period.csv",
    mime="text/csv"
)

per_route_dl = per_route.rename(columns={
    "route":"Ligne", "scheduledTripStops":"Arrêts programmés"
})[["Ligne","Arrêts programmés","Couvert","Entièrement manquant","Autres manquants"]]
st.download_button(
    label="Télécharger – Par ligne (CSV)",
    data=per_route_dl.to_csv(index=False).encode("utf-8"),
    file_name="coverage_by_route.csv",
    mime="text/csv"
)
