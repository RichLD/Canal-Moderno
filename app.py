import streamlit as st
import pandas as pd
import numpy as np

from utils.feature_engineering import procesar_completo
from utils.charts import (
    grafica_yoy,
    grafica_ventas_cadena,
    grafica_abc,
    grafica_inventario,
    grafica_precio_marca,
    grafica_tendencia_vendedor,
    calcular_kpis,
)

st.set_page_config(
    page_title='Sell Out — Arcor',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F5F6FA; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1B2A 0%, #1a2f45 100%);
    border-right: 1px solid #1e3a52;
}
[data-testid="stSidebar"] * { color: #E8ECF0 !important; }
[data-testid="stSidebar"] .stMultiSelect [data-baseweb="tag"] {
    background-color: #F4651E !important; border-radius: 4px; }
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background-color: #1e3a52 !important; border-color: #2d5473 !important; border-radius: 6px; }
[data-testid="stSidebar"] label { font-size: 0.72rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 0.06em; color: #8FA8C0 !important; }
[data-testid="stSidebar"] h2 { color: white !important; font-size: 1.1rem !important; }
[data-testid="stSidebar"] hr { border-color: #2d5473; }
.kpi-wrap { display: flex; flex-direction: column; background: white;
    border-radius: 10px; padding: 1.1rem 1rem; border-top: 3px solid #0D1B2A;
    box-shadow: 0 1px 6px rgba(0,0,0,0.06); height: 100%; }
.kpi-wrap.alert { border-top-color: #F4651E; }
.kpi-wrap.ok    { border-top-color: #27AE60; }
.kpi-label { font-size: 0.68rem; font-weight: 700; color: #8090A0;
    text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 6px; }
.kpi-value { font-size: 1.7rem; font-weight: 700; color: #0D1B2A; line-height: 1; }
.kpi-sub   { font-size: 0.75rem; color: #8090A0; margin-top: 4px; }
.delta-pos { color: #27AE60; font-weight: 600; }
.delta-neg { color: #E74C3C; font-weight: 600; }
.sec-header { font-size: 0.72rem; font-weight: 700; color: #F4651E;
    text-transform: uppercase; letter-spacing: 0.1em;
    border-bottom: 2px solid #F4651E; padding-bottom: 5px; margin: 2rem 0 1rem 0; }
.alert-banner { background: #FFF4EF; border-left: 4px solid #F4651E;
    border-radius: 0 8px 8px 0; padding: 0.8rem 1rem;
    margin-bottom: 1rem; font-size: 0.85rem; color: #0D1B2A; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CARGA CRUDA — solo lee el parquet, sin procesar
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Cargando datos...')
def cargar_raw() -> pd.DataFrame:
    df = pd.read_parquet('BASE/base_slim.parquet', engine='pyarrow')
    df.columns = df.columns.str.strip()
    # Convertir tipos problemáticos a str nativo
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].astype(str).replace('<NA>', '').replace('nan', '')
    for col in df.select_dtypes(include='category').columns:
        df[col] = df[col].astype(str).replace('<NA>', '').replace('nan', '')
    # Tiempo como int para poder filtrar
    df['_año'] = pd.to_numeric(df['Tiempo'], errors='coerce').fillna(0).astype(int) // 100
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — filtros sobre datos crudos
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 📊 Sell Out Arcor')
    st.markdown('---')

    df_raw = cargar_raw()
    st.success(f'✅ {len(df_raw):,} registros')
    st.markdown('---')
    st.markdown('### Filtros')

    años_disp  = sorted(df_raw['_año'].unique().tolist(), reverse=True)
    años_disp  = [a for a in años_disp if a > 0]
    años_sel   = st.multiselect('Año', años_disp, default=años_disp)

    cadenas_disp = sorted(df_raw['Cadena'].unique().tolist())
    cadenas_sel  = st.multiselect('Cadena', cadenas_disp, default=cadenas_disp)

    marcas_disp  = sorted(df_raw['MARCA'].unique().tolist())
    marcas_sel   = st.multiselect('Marca', marcas_disp, default=marcas_disp)

    estados_disp = sorted(df_raw['Estado'].unique().tolist())
    estados_sel  = st.multiselect('Estado', estados_disp, default=estados_disp)

    st.markdown('---')
    st.caption('Arcor México © 2026')


# ─────────────────────────────────────────────────────────────────────────────
# FILTRAR PRIMERO — luego procesar solo el subconjunto
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Procesando...')
def filtrar_y_procesar(años, cadenas, marcas, estados):
    mask = (
        df_raw['_año'].isin(años) &
        df_raw['Cadena'].isin(cadenas) &
        df_raw['MARCA'].isin(marcas) &
        df_raw['Estado'].isin(estados)
    )
    df_filtrado = df_raw[mask].drop(columns=['_año']).copy()
    return procesar_completo(df_filtrado)


datos = filtrar_y_procesar(
    tuple(años_sel), tuple(cadenas_sel),
    tuple(marcas_sel), tuple(estados_sel)
)
base = datos['base']

if base.empty:
    st.warning('⚠️ No hay datos con los filtros seleccionados.')
    st.stop()

kpis    = calcular_kpis(base)
año_max = kpis['año_max']
sem_max = base[base['AÑO'] == año_max]['SEMANA'].max()


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
c1, c2 = st.columns([3, 1])
with c1:
    st.markdown('# 📊 Dashboard Sell Out — Arcor')
    st.caption(f'Última semana: **{año_max} — S{sem_max:02d}** &nbsp;|&nbsp; '
               f'{kpis["tiendas"]:,} tiendas &nbsp;|&nbsp; {base["Cadena"].nunique()} cadenas')
with c2:
    if kpis['var_yoy'] is not None:
        signo = '+' if kpis['var_yoy'] >= 0 else ''
        color = '#27AE60' if kpis['var_yoy'] >= 0 else '#E74C3C'
        st.markdown(f"""
        <div style="text-align:right;padding-top:1rem;">
            <span style="font-size:0.75rem;color:#8090A0;font-weight:700;">YoY VENTA</span><br>
            <span style="font-size:2rem;font-weight:700;color:{color};">{signo}{kpis['var_yoy']:.1f}%</span>
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# BANNER ALERTAS
# ─────────────────────────────────────────────────────────────────────────────
agotados_cnt   = int(base[base['Es_Agotado'] == 1]['NomTienda'].nunique())
agotados_venta = base[base['Es_Agotado'] == 1]['Venta Sell Out Importe Venta'].sum()
exceso_cnt     = int(base['Exceso_Stock'].sum())

if agotados_cnt > 0:
    st.markdown(f"""
    <div class="alert-banner">
        🔴 <b>{agotados_cnt:,} tiendas con agotados</b> que generaron
        <b>${agotados_venta/1e6:.1f}M</b> antes de agotarse &nbsp;|&nbsp;
        ⚠️ <b>{exceso_cnt:,} registros con exceso de stock</b> (&gt;42 días)
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Indicadores Clave</div>', unsafe_allow_html=True)

def kpi_card(col, label, value, sub='', clase=''):
    with col:
        st.markdown(f"""
        <div class="kpi-wrap {clase}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            <div class="kpi-sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

k1, k2, k3, k4, k5, k6 = st.columns(6)
delta_html = ''
if kpis['var_yoy'] is not None:
    cls = 'delta-pos' if kpis['var_yoy'] >= 0 else 'delta-neg'
    signo = '+' if kpis['var_yoy'] >= 0 else ''
    delta_html = f'<span class="{cls}">{signo}{kpis["var_yoy"]:.1f}% vs año ant.</span>'

kpi_card(k1, f'Venta Total {año_max}', f'${kpis["venta_actual"]/1e6:.1f}M', delta_html)
cls_ag = 'alert' if kpis['pct_agotado'] > 8 else 'ok'
kpi_card(k2, '% Agotados', f'{kpis["pct_agotado"]:.1f}%', 'registros sin OH', cls_ag)
dias = kpis['dias_inventario']
cls_d = 'alert' if dias > 30 or dias < 7 else 'ok'
kpi_card(k3, 'Días Cobertura', f'{dias:.0f} días', 'meta: 7–30 días', cls_d)
kpi_card(k4, 'Sell Through', f'{kpis["sell_through"]:.1f}%', 'ventas / (ventas + OH)')
kpi_card(k5, 'Tiendas', f'{kpis["tiendas"]:,}', '')
kpi_card(k6, 'SKUs', f'{kpis["skus"]:,}', '')


# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    '🚨 Oportunidades', '📈 Ventas', '🏷 Portafolio ABC', '📦 Inventario'
])

with tab1:
    st.markdown('<div class="sec-header">SKUs Agotados con Historial de Venta</div>', unsafe_allow_html=True)
    ag_df = (
        base[(base['Es_Agotado'] == 1) & (base['AÑO'] == año_max)]
        .groupby(['Cadena', 'NomTienda', 'Descripcion', 'Estado'])
        .agg(Venta_Piezas=('Venta Sell Out Piezas', 'sum'),
             Venta_Importe=('Venta Sell Out Importe Venta', 'sum'))
        .reset_index()
    )
    ag_df = ag_df[ag_df['Venta_Piezas'] > 0].sort_values('Venta_Importe', ascending=False)
    ag_df['Venta_Importe'] = ag_df['Venta_Importe'].map('${:,.0f}'.format)
    ag_df.columns = ['Cadena', 'Tienda', 'Producto', 'Estado', 'Piezas', 'Importe']
    st.dataframe(ag_df, use_container_width=True, height=350)

    st.markdown('<div class="sec-header">Exceso de Stock (&gt;42 días)</div>', unsafe_allow_html=True)
    ex_df = (
        base[(base['Exceso_Stock'] == 1) & (base['AÑO'] == año_max)]
        .groupby(['Cadena', 'NomTienda', 'Descripcion'])
        .agg(Dias=('Dias_Inventario', 'mean'), OH=('OH Piezas', 'sum'))
        .reset_index()
        .sort_values('Dias', ascending=False)
    )
    ex_df['Dias'] = ex_df['Dias'].map('{:.0f} días'.format)
    ex_df['OH'] = ex_df['OH'].map('{:,.0f}'.format)
    ex_df.columns = ['Cadena', 'Tienda', 'Producto', 'Días Cobertura', 'OH Total']
    st.dataframe(ex_df, use_container_width=True, height=300)

with tab2:
    st.plotly_chart(grafica_yoy(base), use_container_width=True)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(grafica_tendencia_vendedor(base), use_container_width=True)
    with c2:
        st.plotly_chart(grafica_ventas_cadena(base), use_container_width=True)

with tab3:
    from utils.feature_engineering import calcular_abc
    abc_f = calcular_abc(base)
    c1, c2 = st.columns([3, 2])
    with c1:
        st.plotly_chart(grafica_abc(abc_f), use_container_width=True)
    with c2:
        st.plotly_chart(grafica_precio_marca(base), use_container_width=True)
    with st.expander('📋 Tabla completa ABC'):
        df_show = abc_f[['Descripcion', 'UPC', 'Venta Sell Out Importe Venta', 'Perc_Acum', 'Categoria_ABC']].copy()
        df_show['Venta Sell Out Importe Venta'] = df_show['Venta Sell Out Importe Venta'].map('${:,.0f}'.format)
        df_show['Perc_Acum'] = (df_show['Perc_Acum'] * 100).map('{:.1f}%'.format)
        df_show.columns = ['Producto', 'UPC', 'Venta Total', '% Acumulado', 'ABC']
        st.dataframe(df_show, use_container_width=True, height=350)

with tab4:
    st.plotly_chart(grafica_inventario(base), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# DESCARGA
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="sec-header">Exportar</div>', unsafe_allow_html=True)
c1, c2 = st.columns(2)
with c1:
    st.download_button('⬇️ Base procesada (.csv)',
                       base.to_csv(index=False).encode('utf-8'),
                       'base_procesada.csv', 'text/csv')
with c2:
    from utils.feature_engineering import calcular_abc as _abc
    st.download_button('⬇️ Clasificación ABC (.csv)',
                       _abc(base).to_csv(index=False).encode('utf-8'),
                       'abc.csv', 'text/csv')
