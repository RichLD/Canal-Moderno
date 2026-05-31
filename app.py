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

# ─────────────────────────────────────────────────────────────────────────────
# Configuración de página
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title='Dashboard Sell Out — Arcor',
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded',
)

# CSS personalizado
st.markdown("""
<style>
    /* Fondo general */
    .main { background-color: #F8F9FA; }
    .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 10px;
        padding: 1.2rem 1rem;
        text-align: center;
        border-left: 4px solid #1B2A4A;
        box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    }
    .kpi-card.alerta { border-left-color: #E8501A; }
    .kpi-card.ok     { border-left-color: #27AE60; }
    .kpi-label { font-size: 0.72rem; color: #6C757D; font-weight: 600;
                 text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; }
    .kpi-value { font-size: 1.55rem; font-weight: 700; color: #1B2A4A; }
    .kpi-delta { font-size: 0.8rem; margin-top: 3px; }
    .delta-pos { color: #27AE60; }
    .delta-neg { color: #E74C3C; }

    /* Sección headers */
    .section-header {
        font-size: 1rem; font-weight: 700; color: #1B2A4A;
        border-bottom: 2px solid #E8501A;
        padding-bottom: 4px; margin: 1.5rem 0 1rem 0;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #1B2A4A; }
    [data-testid="stSidebar"] * { color: white !important; }
    [data-testid="stSidebar"] .stMultiSelect > div { border-color: #ffffff44; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Caché de procesamiento
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner='Cargando base de datos...')
def cargar_datos() -> dict:
    df_raw = pd.read_parquet('BASE/base_optimizada.parquet')
    df_raw.columns = df_raw.columns.str.strip()
    return procesar_completo(df_raw)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Filtros
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('## 📊 Dashboard Sell Out')
    st.markdown('---')

    datos = cargar_datos()
    base_full = datos['base']

    st.success(f'✅ {len(base_full):,} registros cargados')
    st.markdown('---')
    st.markdown('### Filtros')

    # Año
    años_disp = sorted(base_full['AÑO'].dropna().unique().tolist(), reverse=True)
    años_sel = st.multiselect('Año', años_disp, default=años_disp[:2])

    # Cadena
    cadenas_disp = sorted(base_full['Cadena'].dropna().unique().tolist())
    cadenas_sel = st.multiselect('Cadena', cadenas_disp, default=cadenas_disp)

    # Vendedor
    vendedores_disp = sorted(base_full['Vendedores'].dropna().unique().tolist())
    vendedores_sel = st.multiselect('Vendedor', vendedores_disp, default=vendedores_disp)

    # Marca
    marcas_disp = sorted(base_full['MARCA'].dropna().unique().tolist())
    marcas_sel = st.multiselect('Marca', marcas_disp, default=marcas_disp)

    # Estado
    estados_disp = sorted(base_full['Estado'].dropna().unique().tolist())
    estados_sel = st.multiselect('Estado', estados_disp, default=estados_disp)

    st.markdown('---')
    st.caption('Arcor México • Sell Out Analytics')


# ─────────────────────────────────────────────────────────────────────────────
# Aplicar filtros
# ─────────────────────────────────────────────────────────────────────────────
mask = (
    base_full['AÑO'].isin(años_sel) &
    base_full['Cadena'].isin(cadenas_sel) &
    base_full['Vendedores'].isin(vendedores_sel) &
    base_full['MARCA'].isin(marcas_sel) &
    base_full['Estado'].isin(estados_sel)
)
base = base_full[mask].copy()

if base.empty:
    st.warning('⚠️ No hay datos con los filtros seleccionados.')
    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# TÍTULO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('# 📊 Dashboard Sell Out — Arcor')
año_max = base['AÑO'].max()
semana_max = base[base['AÑO'] == año_max]['SEMANA'].max()
st.caption(f'Última semana disponible: **{año_max} — S{semana_max:02d}** &nbsp;|&nbsp; '
           f'{base["NomTienda"].nunique():,} tiendas &nbsp;|&nbsp; '
           f'{base["Cadena"].nunique()} cadenas')


# ─────────────────────────────────────────────────────────────────────────────
# KPIs
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">KPIs Generales</div>', unsafe_allow_html=True)

kpis = calcular_kpis(base)

col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    delta_html = ''
    if kpis['var_yoy'] is not None:
        cls = 'delta-pos' if kpis['var_yoy'] >= 0 else 'delta-neg'
        signo = '+' if kpis['var_yoy'] >= 0 else ''
        delta_html = f'<div class="kpi-delta"><span class="{cls}">{signo}{kpis["var_yoy"]:.1f}% vs año ant.</span></div>'
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Venta Total {kpis['año_max']}</div>
        <div class="kpi-value">${kpis['venta_actual']/1e6:.1f}M</div>
        {delta_html}
    </div>""", unsafe_allow_html=True)

with col2:
    cls_inv = 'alerta' if kpis['pct_agotado'] > 10 else 'ok'
    st.markdown(f"""
    <div class="kpi-card {cls_inv}">
        <div class="kpi-label">% Agotados</div>
        <div class="kpi-value">{kpis['pct_agotado']:.1f}%</div>
        <div class="kpi-delta">de registros sin OH</div>
    </div>""", unsafe_allow_html=True)

with col3:
    dias = kpis['dias_inventario']
    cls_dias = 'alerta' if dias > 30 or dias < 7 else 'ok'
    st.markdown(f"""
    <div class="kpi-card {cls_dias}">
        <div class="kpi-label">Días Cobertura Prom.</div>
        <div class="kpi-value">{dias:.0f} días</div>
        <div class="kpi-delta">meta: 7–30 días</div>
    </div>""", unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Sell Through Rate</div>
        <div class="kpi-value">{kpis['sell_through']:.1f}%</div>
        <div class="kpi-delta">ventas / (ventas + OH)</div>
    </div>""", unsafe_allow_html=True)

with col5:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Tiendas Activas</div>
        <div class="kpi-value">{kpis['tiendas']:,}</div>
    </div>""", unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">SKUs</div>
        <div class="kpi-value">{kpis['skus']:,}</div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 1 — Ventas
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Ventas Sell Out</div>', unsafe_allow_html=True)

col_yoy, col_cadena = st.columns([2, 1])
with col_yoy:
    st.plotly_chart(grafica_yoy(base), use_container_width=True)
with col_cadena:
    st.plotly_chart(grafica_ventas_cadena(base), use_container_width=True)

st.plotly_chart(grafica_tendencia_vendedor(base), use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 2 — Inventario
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Salud de Inventario</div>', unsafe_allow_html=True)
st.plotly_chart(grafica_inventario(base), use_container_width=True)

# Tabla de alertas: agotados con venta
with st.expander('🔴 Detalle de SKUs Agotados con Venta Reciente'):
    sem_ref = base['SEMANA'].max()
    agotados = (
        base[(base['Es_Agotado'] == 1) & (base['AÑO'] == año_max)]
        .groupby(['Cadena', 'NomTienda', 'Descripcion', 'Estado'])
        .agg(
            OH_Piezas=('OH Piezas', 'mean'),
            Venta_Piezas=('Venta Sell Out Piezas', 'sum'),
        )
        .reset_index()
    )
    agotados = agotados[agotados['Venta_Piezas'] > 0].sort_values('Venta_Piezas', ascending=False)
    st.dataframe(agotados, use_container_width=True, height=300)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 3 — Portafolio
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Portafolio — ABC y Precios</div>', unsafe_allow_html=True)

# Recalcular ABC sobre datos filtrados
from utils.feature_engineering import calcular_abc
abc_filtrado = calcular_abc(base)

col_abc, col_precio = st.columns([3, 2])
with col_abc:
    st.plotly_chart(grafica_abc(abc_filtrado), use_container_width=True)
with col_precio:
    st.plotly_chart(grafica_precio_marca(base), use_container_width=True)

# Tabla ABC completa
with st.expander('📋 Tabla completa ABC'):
    df_show = abc_filtrado[['Descripcion', 'UPC', 'Venta Sell Out Importe Venta',
                             'Perc_Acum', 'Categoria_ABC']].copy()
    df_show['Venta Sell Out Importe Venta'] = df_show['Venta Sell Out Importe Venta'].map('${:,.0f}'.format)
    df_show['Perc_Acum'] = (df_show['Perc_Acum'] * 100).map('{:.1f}%'.format)
    df_show.columns = ['Producto', 'UPC', 'Venta Total', '% Acumulado', 'ABC']
    st.dataframe(df_show, use_container_width=True, height=350)


# ─────────────────────────────────────────────────────────────────────────────
# SECCIÓN 4 — Descarga
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Exportar Datos</div>', unsafe_allow_html=True)

col_d1, col_d2 = st.columns(2)

with col_d1:
    csv_base = base.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Descargar base procesada (.csv)',
        data=csv_base,
        file_name='base_procesada.csv',
        mime='text/csv',
    )

with col_d2:
    csv_abc = abc_filtrado.to_csv(index=False).encode('utf-8')
    st.download_button(
        label='⬇️ Descargar clasificación ABC (.csv)',
        data=csv_abc,
        file_name='abc.csv',
        mime='text/csv',
    )
