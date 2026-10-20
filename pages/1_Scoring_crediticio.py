import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

st.set_page_config(page_title="Scoring crediticio | Macro Perú", layout="wide")

LABELS = {
    "edad": "Edad",
    "ingreso_mensual": "Ingreso mensual (S/)",
    "antiguedad_anios": "Antigüedad laboral (años)",
    "monto": "Monto del préstamo (S/)",
    "plazo_meses": "Plazo (meses)",
    "deuda_otros_ratio": "Deuda en otras entidades / ingreso",
    "uso_linea": "Uso de línea de crédito",
    "atrasos_12m": "Atrasos últimos 12 meses",
    "ratio_cuota_ingreso": "Cuota / ingreso",
}
FEATS = list(LABELS)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


@st.cache_data
def make_data(n: int = 20000, seed: int = 42) -> pd.DataFrame:
    """Datos sintéticos: no son clientes reales."""
    rng = np.random.default_rng(seed)
    edad = rng.integers(21, 66, n)
    ingreso = np.round(rng.lognormal(7.6, 0.5, n), -1)
    antig = np.minimum(np.clip(rng.normal(5, 4, n), 0, 40), edad - 18).round(1)
    plazo = rng.choice([12, 24, 36, 48, 60], n)
    monto = np.round(ingreso * rng.uniform(2, 12, n), -2)
    deuda_otros = (rng.uniform(0, 3, n)).round(2)
    uso = rng.beta(2, 3, n).round(2)
    atrasos = np.clip(rng.poisson(0.5 + 1.2 * uso), 0, 8)
    ratio = (monto / plazo) / ingreso
    z = (-3.9 + 3.2 * ratio + 0.55 * atrasos + 1.4 * uso + 0.35 * deuda_otros
         - 0.05 * antig - 0.012 * (edad - 40) - 0.00008 * (ingreso - 2000)
         + rng.normal(0, 0.4, n))
    default = rng.binomial(1, sigmoid(z))
    return pd.DataFrame({
        "edad": edad, "ingreso_mensual": ingreso, "antiguedad_anios": antig,
        "monto": monto, "plazo_meses": plazo, "deuda_otros_ratio": deuda_otros,
        "uso_linea": uso, "atrasos_12m": atrasos, "ratio_cuota_ingreso": ratio.round(3),
        "default": default,
    })


@st.cache_resource(show_spinner="Entrenando modelo…")
def train():
    df = make_data()
    Xtr, Xte, ytr, yte = train_test_split(df[FEATS], df["default"], test_size=0.25,
                                          random_state=1, stratify=df["default"])
    model = XGBClassifier(n_estimators=250, max_depth=3, learning_rate=0.07,
                          subsample=0.8, colsample_bytree=0.9, eval_metric="logloss")
    model.fit(Xtr, ytr)
    auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])
    return model, shap.TreeExplainer(model), Xte.reset_index(drop=True), auc


model, explainer, Xte, auc = train()

st.title("Scoring crediticio explicable")
st.caption("Modelo XGBoost + SHAP sobre datos sintéticos, con estrés macro (tasa de referencia y tipo de cambio).")

# ---------- Escenario macro (sidebar) ----------
with st.sidebar:
    st.header("Escenario macro")
    d_tasa = st.slider("Cambio en tasa de referencia (pp)", -2.0, 4.0, 0.0, 0.25)
    deval = st.slider("Depreciación del sol (%)", -5.0, 25.0, 0.0, 0.5)
    st.caption("Supuesto: +1 pp de tasa suma 0.18 y +1% de depreciación suma 0.04 al log-odds de default. "
               "Parámetros ilustrativos, calibrables con datos reales.")
    umbral = st.slider("Umbral de aprobación (PD máx.)", 0.02, 0.40, 0.12, 0.01)

shift = 0.18 * d_tasa + 0.04 * deval

tab1, tab2, tab3 = st.tabs(["Simulador", "Portafolio", "Modelo"])

# ---------- Simulador ----------
with tab1:
    c1, c2, c3 = st.columns(3)
    edad = c1.number_input("Edad", 18, 75, 35)
    ingreso = c1.number_input("Ingreso mensual (S/)", 800, 30000, 3000, 100)
    antig = c1.number_input("Antigüedad laboral (años)", 0.0, 40.0, 4.0, 0.5)
    monto = c2.number_input("Monto del préstamo (S/)", 1000, 200000, 15000, 500)
    plazo = c2.selectbox("Plazo (meses)", [12, 24, 36, 48, 60], index=2)
    deuda = c2.slider("Deuda en otras entidades / ingreso", 0.0, 3.0, 0.5, 0.05)
    uso = c3.slider("Uso de línea de crédito", 0.0, 1.0, 0.4, 0.05)
    atr = c3.number_input("Atrasos últimos 12 meses", 0, 8, 0)

    ratio = (monto / plazo) / ingreso
    row = pd.DataFrame([[edad, ingreso, antig, monto, plazo, deuda, uso, atr, round(ratio, 3)]], columns=FEATS)
    pd_base = float(model.predict_proba(row)[0, 1])
    pd_adj = float(sigmoid(logit(pd_base) + shift))

    k1, k2, k3 = st.columns(3)
    k1.metric("PD base", f"{pd_base:.1%}")
    k2.metric("PD con escenario macro", f"{pd_adj:.1%}", f"{(pd_adj - pd_base) * 100:+.1f} pp", delta_color="inverse")
    k3.metric("Decisión", "Aprobar" if pd_adj <= umbral else "Rechazar")

    sv = explainer.shap_values(row)[0]
    contrib = pd.Series(sv, index=[LABELS[f] for f in FEATS])
    if shift != 0:
        contrib["Escenario macro"] = shift
    contrib = contrib.reindex(contrib.abs().sort_values().index)
    fig = go.Figure(go.Bar(x=contrib.values, y=contrib.index, orientation="h",
                           marker_color=["#dc2626" if v > 0 else "#16a34a" for v in contrib.values]))
    fig.update_layout(template="plotly_white", height=420, margin=dict(l=10, r=10, t=30, b=10),
                      title="Qué empuja el riesgo (rojo sube, verde baja) · log-odds")
    st.plotly_chart(fig, use_container_width=True)

# ---------- Portafolio ----------
with tab2:
    st.subheader("Pérdida esperada: base vs. escenario macro")
    lgd = st.slider("LGD (pérdida dado el default)", 0.2, 0.9, 0.45, 0.05)
    port = Xte.copy()
    port["PD base"] = model.predict_proba(port[FEATS])[:, 1]
    port["PD escenario"] = sigmoid(logit(port["PD base"]) + shift)
    port["EL base"] = port["PD base"] * lgd * port["monto"]
    port["EL escenario"] = port["PD escenario"] * lgd * port["monto"]
    port["Tramo de ingreso"] = pd.cut(port["ingreso_mensual"], [0, 1500, 3000, 6000, 1e9],
                                      labels=["<1,500", "1,500–3,000", "3,000–6,000", ">6,000"])
    port["Plazo"] = port["plazo_meses"].astype(str) + " m"
    port["Atrasos"] = pd.cut(port["atrasos_12m"], [-1, 0, 2, 99], labels=["0", "1–2", "3+"])

    f1, f2, f3 = st.columns(3)
    tr = f1.multiselect("Tramo de ingreso", list(port["Tramo de ingreso"].cat.categories),
                        default=list(port["Tramo de ingreso"].cat.categories))
    pl = f2.multiselect("Plazo", sorted(port["Plazo"].unique(), key=lambda s: int(s.split()[0])),
                        default=sorted(port["Plazo"].unique(), key=lambda s: int(s.split()[0])))
    grupo = f3.selectbox("Agrupar por", ["Tramo de ingreso", "Plazo", "Atrasos"])
    sub = port[port["Tramo de ingreso"].isin(tr) & port["Plazo"].isin(pl)]

    g = sub.groupby(grupo, observed=True).agg(
        Clientes=("monto", "size"), Exposición=("monto", "sum"),
        PD_base=("PD base", "mean"), PD_escenario=("PD escenario", "mean"),
        EL_base=("EL base", "sum"), EL_escenario=("EL escenario", "sum"))
    g["Δ EL"] = g["EL_escenario"] - g["EL_base"]
    g["Δ EL %"] = g["Δ EL"] / g["EL_base"] * 100
    st.dataframe(
        g.style.format({"Exposición": "{:,.0f}", "PD_base": "{:.1%}", "PD_escenario": "{:.1%}",
                        "EL_base": "{:,.0f}", "EL_escenario": "{:,.0f}", "Δ EL": "{:+,.0f}", "Δ EL %": "{:+.1f}%"})
         .background_gradient(cmap="Reds", subset=["Δ EL"]),
        use_container_width=True)
    st.download_button("Descargar CSV", g.to_csv().encode(), "perdida_esperada.csv", "text/csv")
    tot_b, tot_e = sub["EL base"].sum(), sub["EL escenario"].sum()
    st.metric("Pérdida esperada total (S/)", f"{tot_e:,.0f}", f"{tot_e - tot_b:+,.0f} vs base", delta_color="inverse")

# ---------- Modelo ----------
with tab3:
    st.metric("AUC en test", f"{auc:.3f}")
    imp = pd.Series(np.abs(explainer.shap_values(Xte.sample(1500, random_state=0))).mean(0),
                    index=[LABELS[f] for f in FEATS]).sort_values()
    figi = go.Figure(go.Bar(x=imp.values, y=imp.index, orientation="h", marker_color="#2563eb"))
    figi.update_layout(template="plotly_white", height=380, margin=dict(l=10, r=10, t=30, b=10),
                       title="Importancia global (|SHAP| medio)")
    st.plotly_chart(figi, use_container_width=True)
    st.caption("Los datos son sintéticos y el efecto macro es un supuesto ilustrativo. "
               "Con datos reales, el mismo código sirve reemplazando make_data().")
