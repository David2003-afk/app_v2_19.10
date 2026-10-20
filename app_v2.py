import re
import warnings
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
st.set_page_config(page_title="Macro Perú | Pronóstico BCRP", layout="wide")

# ---------- Configuración ----------
SERIES = {
    "Tipo de cambio (S/ por US$)": {"code": "PD04638PD", "freq": "D", "agg": "mean"},
    "IPC Lima (índice 2009=100)": {"code": "PN01271PM", "freq": "M", "agg": "last"},
    "Tasa de referencia BCRP (%)": {"code": "PN07819NM", "freq": "M", "agg": "last"},
}
MESES = {"Ene": 1, "Feb": 2, "Mar": 3, "Abr": 4, "May": 5, "Jun": 6, "Jul": 7,
         "Ago": 8, "Set": 9, "Sep": 9, "Oct": 10, "Nov": 11, "Dic": 12}
MES_LABEL = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"


def parse_period(name: str, freq: str):
    if freq == "D":  # ej. 02Ene20
        m = re.match(r"(\d{2})([A-Za-z]{3})(\d{2})", name)
        return pd.Timestamp(2000 + int(m[3]), MESES[m[2].capitalize()], int(m[1]))
    m = re.match(r"([A-Za-z]{3})\.?(\d{4})", name)  # ej. Ene.2020
    return pd.Timestamp(int(m[2]), MESES[m[1].capitalize()], 1)


@st.cache_data(ttl=6 * 3600, show_spinner="Consultando BCRPData…")
def load_series(code: str, freq: str, start_year: int = 2010) -> pd.Series:
    today = date.today()
    if freq == "D":
        rng = f"{start_year}-1-1/{today.year}-{today.month}-{today.day}"
    else:
        rng = f"{start_year}-1/{today.year}-{today.month}"
    r = requests.get(f"{BASE}/{code}/json/{rng}/esp", timeout=40)
    r.raise_for_status()
    rows = r.json()["periods"]
    idx, vals = [], []
    for p in rows:
        try:
            v = float(p["values"][0])
        except (ValueError, TypeError):
            continue  # 'n.d.'
        idx.append(parse_period(p["name"], freq))
        vals.append(v)
    s = pd.Series(vals, index=pd.DatetimeIndex(idx)).sort_index()
    return s


def to_monthly(s: pd.Series, agg: str) -> pd.Series:
    m = s.resample("MS").mean() if agg == "mean" else s.resample("MS").last()
    return m.dropna()


@st.cache_data(show_spinner="Ajustando modelos…")
def forecast(s: pd.Series, horizon: int, model: str) -> pd.DataFrame:
    s = s.asfreq("MS").interpolate()
    if model == "SARIMA":
        res = SARIMAX(s, order=(1, 1, 1), seasonal_order=(1, 0, 0, 12),
                      enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
        fc = res.get_forecast(horizon)
        ci = fc.conf_int(alpha=0.05)
        return pd.DataFrame({"pronostico": fc.predicted_mean,
                             "inferior": ci.iloc[:, 0], "superior": ci.iloc[:, 1]})
    # Holt-Winters amortiguado con bandas por residuos
    res = ExponentialSmoothing(s, trend="add", damped_trend=True).fit()
    pred = res.forecast(horizon)
    sigma = np.std(res.resid)
    h = np.arange(1, horizon + 1)
    band = 1.96 * sigma * np.sqrt(h)
    return pd.DataFrame({"pronostico": pred, "inferior": pred - band, "superior": pred + band})


# ---------- Sidebar ----------
st.title("Macro Perú: tipo de cambio, inflación y tasa de referencia")
st.caption("Datos oficiales del BCRP (BCRPData API) · Pronóstico con bandas de confianza al 95%")

with st.sidebar:
    st.header("Filtros")
    var = st.selectbox("Variable", list(SERIES))
    desde = st.slider("Desde el año", 2010, date.today().year - 1, 2018)
    modelo = st.radio("Modelo de pronóstico", ["SARIMA", "Holt-Winters", "Comparar ambos"])
    horizonte = st.slider("Horizonte (meses)", 3, 36, 12)

cfg = SERIES[var]
try:
    raw = load_series(cfg["code"], cfg["freq"])
except Exception as e:
    st.error(f"No se pudo consultar el BCRP: {e}")
    st.stop()

mensual = to_monthly(raw, cfg["agg"])
mensual = mensual[mensual.index.year >= desde]

# KPIs
ult, ant = mensual.iloc[-1], mensual.iloc[-2]
yoy = mensual.iloc[-1] / mensual.iloc[-13] - 1 if len(mensual) > 12 else np.nan
c1, c2, c3 = st.columns(3)
c1.metric("Último dato", f"{ult:,.3f}", f"{ult - ant:+.3f} vs mes anterior")
c2.metric("Variación 12 meses", f"{yoy:.2%}" if pd.notna(yoy) else "n.d.")
c3.metric("Fecha", mensual.index[-1].strftime("%b %Y"))

tab1, tab2, tab3, tab4 = st.tabs(["Pronóstico", "Tabla dinámica", "Comparar períodos", "Escenarios"])

# ---------- Tab 1: pronóstico ----------
with tab1:
    modelos = ["SARIMA", "Holt-Winters"] if modelo == "Comparar ambos" else [modelo]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=mensual.index, y=mensual, name="Histórico", line=dict(color="#1f2937")))
    colores = {"SARIMA": "#2563eb", "Holt-Winters": "#dc2626"}
    fcs = {}
    for m in modelos:
        fc = forecast(mensual, horizonte, m)
        fcs[m] = fc
        col = colores[m]
        fig.add_trace(go.Scatter(x=fc.index, y=fc["superior"], line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["inferior"], fill="tonexty", line=dict(width=0),
                                 fillcolor="rgba(37,99,235,0.12)" if m == "SARIMA" else "rgba(220,38,38,0.12)",
                                 name=f"Banda 95% {m}"))
        fig.add_trace(go.Scatter(x=fc.index, y=fc["pronostico"], name=f"Pronóstico {m}",
                                 line=dict(color=col, dash="dash")))
    fig.update_layout(template="plotly_white", height=470, hovermode="x unified",
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, use_container_width=True)
    st.session_state["fcs"] = fcs

# ---------- Tab 2: tabla dinámica ----------
with tab2:
    st.subheader("Tabla dinámica año × mes")
    a, b, c = st.columns(3)
    anios = sorted(mensual.index.year.unique())
    sel = a.multiselect("Años", anios, default=anios[-5:])
    agg = b.selectbox("Agregación", ["mean", "min", "max", "last"], format_func=lambda x: {
        "mean": "Promedio", "min": "Mínimo", "max": "Máximo", "last": "Último"}[x])
    modo = c.selectbox("Mostrar", ["Nivel", "Variación % vs mes anterior", "Variación % interanual"])

    df = mensual.to_frame("valor")
    if modo == "Variación % vs mes anterior":
        df["valor"] = df["valor"].pct_change() * 100
    elif modo == "Variación % interanual":
        df["valor"] = df["valor"].pct_change(12) * 100
    df["Año"], df["Mes"] = df.index.year, df.index.month
    df = df[df["Año"].isin(sel)]
    pv = df.pivot_table(index="Año", columns="Mes", values="valor", aggfunc=agg)
    pv.columns = [MES_LABEL[m - 1] for m in pv.columns]
    st.dataframe(pv.style.format("{:.3f}").background_gradient(cmap="Blues", axis=None),
                 use_container_width=True)
    st.download_button("Descargar CSV", pv.to_csv().encode(), "tabla_dinamica.csv", "text/csv")

# ---------- Tab 3: comparar ----------
with tab3:
    st.subheader("Diferencia entre dos períodos")
    ano_a, ano_b = st.columns(2)
    anios_all = sorted(to_monthly(raw, cfg["agg"]).index.year.unique())
    p1 = ano_a.selectbox("Año base", anios_all, index=max(len(anios_all) - 2, 0))
    p2 = ano_b.selectbox("Año a comparar", anios_all, index=len(anios_all) - 1)
    full = to_monthly(raw, cfg["agg"])
    t = pd.DataFrame({
        str(p1): full[full.index.year == p1].groupby(full[full.index.year == p1].index.month).mean(),
        str(p2): full[full.index.year == p2].groupby(full[full.index.year == p2].index.month).mean(),
    })
    t["Diferencia"] = t[str(p2)] - t[str(p1)]
    t["Dif. %"] = t["Diferencia"] / t[str(p1)] * 100
    t.index = [MES_LABEL[i - 1] for i in t.index]
    st.dataframe(t.style.format("{:.3f}").background_gradient(cmap="RdYlGn_r", subset=["Dif. %"]),
                 use_container_width=True)
    figc = go.Figure()
    figc.add_bar(x=t.index, y=t["Diferencia"], marker_color="#2563eb", name="Diferencia")
    figc.update_layout(template="plotly_white", height=300, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(figc, use_container_width=True)

# ---------- Tab 4: escenarios ----------
with tab4:
    st.subheader("Pronóstico por modelo y escenario")
    fcs = st.session_state.get("fcs", {})
    if not fcs:
        st.info("Genera un pronóstico en la primera pestaña.")
    else:
        choque = st.slider("Choque sobre el pronóstico (%)", -10.0, 10.0, 0.0, 0.5)
        out = pd.DataFrame(index=next(iter(fcs.values())).index)
        for m, f in fcs.items():
            out[f"{m} base"] = f["pronostico"]
            out[f"{m} escenario"] = f["pronostico"] * (1 + choque / 100)
            out[f"{m} inf 95%"] = f["inferior"]
            out[f"{m} sup 95%"] = f["superior"]
        if len(fcs) == 2:
            out["Dif. SARIMA vs HW"] = out["SARIMA base"] - out["Holt-Winters base"]
        out.index = out.index.strftime("%b %Y")
        st.dataframe(out.style.format("{:.3f}"), use_container_width=True)
        st.download_button("Descargar pronóstico", out.to_csv().encode(), "pronostico.csv", "text/csv")

st.divider()
st.caption("Fuente: BCRPData. Los pronósticos son estadísticos y no constituyen asesoría financiera.")
