import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PALETTE = {
    'primary': '#0D1B2A',
    'accent':  '#F4651E',
    'accent2': '#F39C12',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'neutral':  '#95A5A6',
    'grid':     '#E9ECEF',
}


def _layout(title: str) -> dict:
    return dict(
        title=dict(text=f'<b>{title}</b>', font=dict(size=15, color=PALETTE['primary'])),
        plot_bgcolor='white', paper_bgcolor='white',
        font=dict(family='Inter, Arial', color=PALETTE['primary']),
        margin=dict(l=40, r=20, t=55, b=40),
        xaxis=dict(gridcolor=PALETTE['grid'], linecolor=PALETTE['grid']),
        yaxis=dict(gridcolor=PALETTE['grid'], linecolor=PALETTE['grid']),
        hovermode='x unified',
    )


# ── KPIs ─────────────────────────────────────────────────────────────────────

def calcular_kpis(base: pd.DataFrame, df_2025: pd.DataFrame,
                  cadenas_sel: list, vend_sel: list) -> dict:
    venta_actual = pd.to_numeric(base['Venta Sell Out Importe Venta'], errors='coerce').sum()

    # YoY con resumen 2025 filtrado por cadena/vendedor
    df_25 = df_2025.copy()
    if cadenas_sel:
        df_25 = df_25[df_25['Cadena'].isin(cadenas_sel)]
    if vend_sel and 'Vendedores' in df_25.columns:
        df_25 = df_25[df_25['Vendedores'].isin(vend_sel)]
    venta_anterior = pd.to_numeric(df_25.get('Venta_Importe', pd.Series([0])), errors='coerce').sum()
    var_yoy = ((venta_actual / venta_anterior) - 1) * 100 if venta_anterior > 0 else None

    oh = pd.to_numeric(base.get('OH Piezas', 0), errors='coerce').fillna(0)
    pct_agotado = (oh <= 0).mean() * 100 if len(base) > 0 else 0

    dias_col = pd.to_numeric(base.get('Dias_Inventario', pd.Series([])), errors='coerce')
    dias_inv = dias_col[dias_col > 0].mean() if len(dias_col[dias_col > 0]) > 0 else 0

    st_col = pd.to_numeric(base.get('Sell_Through_Rate', pd.Series([])), errors='coerce').dropna()
    sell_through = st_col.mean() * 100 if len(st_col) > 0 else 0

    return {
        'venta_actual': venta_actual,
        'var_yoy': var_yoy,
        'pct_agotado': pct_agotado,
        'dias_inventario': dias_inv,
        'sell_through': sell_through,
    }


# ── YoY ──────────────────────────────────────────────────────────────────────

def grafica_yoy(base: pd.DataFrame, df_2025: pd.DataFrame,
                cadenas_sel: list, vend_sel: list) -> go.Figure:
    # 2026 agregado por semana
    df_26 = (
        base.groupby('SEMANA')['Venta Sell Out Importe Venta']
        .sum().reset_index().sort_values('SEMANA')
    )

    # 2025 filtrado y agregado
    df_25 = df_2025.copy()
    if cadenas_sel:
        df_25 = df_25[df_25['Cadena'].isin(cadenas_sel)]
    if vend_sel and 'Vendedores' in df_25.columns:
        df_25 = df_25[df_25['Vendedores'].isin(vend_sel)]
    df_25 = df_25.groupby('SEMANA')['Venta_Importe'].sum().reset_index().sort_values('SEMANA')

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_25['SEMANA'], y=df_25['Venta_Importe'],
        name='2025', mode='lines',
        line=dict(color=PALETTE['neutral'], dash='dot', width=1.5),
        hovertemplate='<b>2025</b> S%{x}: $%{y:,.0f}<extra></extra>',
    ))
    fig.add_trace(go.Scatter(
        x=df_26['SEMANA'], y=df_26['Venta Sell Out Importe Venta'],
        name='2026', mode='lines+markers',
        line=dict(color=PALETTE['accent'], width=3),
        marker=dict(size=4),
        hovertemplate='<b>2026</b> S%{x}: $%{y:,.0f}<extra></extra>',
    ))
    layout = _layout('Venta Sell Out — Comparativo YoY')
    layout['yaxis']['tickprefix'] = '$'
    layout['xaxis']['title'] = 'Semana'
    layout['legend'] = dict(orientation='h', y=1.1)
    fig.update_layout(**layout)
    return fig


# ── Ventas por Cadena ─────────────────────────────────────────────────────────

def grafica_ventas_cadena(base: pd.DataFrame) -> go.Figure:
    df = (
        base.groupby('Cadena')['Venta Sell Out Importe Venta']
        .sum().sort_values().reset_index()
    )
    total = df['Venta Sell Out Importe Venta'].sum()
    df['Pct'] = df['Venta Sell Out Importe Venta'] / total * 100

    fig = go.Figure(go.Bar(
        x=df['Venta Sell Out Importe Venta'], y=df['Cadena'],
        orientation='h', marker_color=PALETTE['primary'],
        text=df['Pct'].map('{:.1f}%'.format), textposition='outside',
        hovertemplate='<b>%{y}</b><br>$%{x:,.0f}<extra></extra>',
    ))
    layout = _layout('Ventas por Cadena')
    layout['xaxis']['tickprefix'] = '$'
    layout.pop('hovermode')
    fig.update_layout(**layout)
    return fig


# ── ABC ───────────────────────────────────────────────────────────────────────

def grafica_abc(base: pd.DataFrame):
    """Devuelve (figura, tabla_df)."""
    df = (
        base.groupby(['UPC', 'Descripcion'])['Venta Sell Out Importe Venta']
        .sum().sort_values(ascending=False).reset_index()
    )
    total = df['Venta Sell Out Importe Venta'].sum()
    df['Perc_Acum'] = df['Venta Sell Out Importe Venta'].cumsum() / total
    df['ABC'] = pd.cut(df['Perc_Acum'], bins=[0, 0.80, 0.95, 1.0], labels=['A', 'B', 'C'])
    df_top = df.head(40).copy()

    color_map = {'A': PALETTE['accent'], 'B': PALETTE['accent2'], 'C': PALETTE['neutral']}
    colores = df_top['ABC'].astype(str).map(color_map)

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(go.Bar(
        x=df_top['Descripcion'].str[:25],
        y=df_top['Venta Sell Out Importe Venta'],
        marker_color=colores, name='Venta',
        hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>',
    ), secondary_y=False)
    fig.add_trace(go.Scatter(
        x=df_top['Descripcion'].str[:25],
        y=df_top['Perc_Acum'] * 100,
        mode='lines+markers',
        line=dict(color=PALETTE['primary'], width=2),
        name='% Acum.',
        hovertemplate='%{y:.1f}%<extra></extra>',
    ), secondary_y=True)
    fig.update_layout(**_layout('Clasificación ABC — Top Productos'))
    fig.update_yaxes(tickprefix='$', secondary_y=False, gridcolor=PALETTE['grid'])
    fig.update_yaxes(ticksuffix='%', secondary_y=True, gridcolor=PALETTE['grid'])
    fig.update_xaxes(tickangle=-45)

    # Tabla
    tabla = df[['Descripcion', 'UPC', 'Venta Sell Out Importe Venta', 'Perc_Acum', 'ABC']].copy()
    tabla['Venta Sell Out Importe Venta'] = tabla['Venta Sell Out Importe Venta'].map('${:,.0f}'.format)
    tabla['Perc_Acum'] = (tabla['Perc_Acum'] * 100).map('{:.1f}%'.format)
    tabla.columns = ['Producto', 'UPC', 'Venta Total', '% Acumulado', 'ABC']

    return fig, tabla


# ── Inventario ────────────────────────────────────────────────────────────────

def grafica_inventario(base: pd.DataFrame) -> go.Figure:
    df = base.groupby('Cadena').agg(
        Dias=('Dias_Inventario', 'mean'),
        Pct_Ag=('Es_Agotado', 'mean'),
        Pct_Ex=('Exceso_Stock', 'mean'),
    ).reset_index()
    df['Pct_Ag'] *= 100
    df['Pct_Ex'] *= 100
    df = df.sort_values('Dias', ascending=False)

    fig = make_subplots(rows=1, cols=2,
        subplot_titles=('Días Cobertura Promedio', '% Agotados vs Exceso'))
    fig.add_trace(go.Bar(x=df['Cadena'], y=df['Dias'],
        marker_color=PALETTE['primary'], name='Días',
        hovertemplate='<b>%{x}</b><br>%{y:.1f} días<extra></extra>'),
        row=1, col=1)
    fig.add_hline(y=14, line_dash='dash', line_color=PALETTE['accent'],
                  annotation_text='14 días', row=1, col=1)
    fig.add_trace(go.Bar(x=df['Cadena'], y=df['Pct_Ag'],
        name='% Agotados', marker_color=PALETTE['negative'],
        hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>'), row=1, col=2)
    fig.add_trace(go.Bar(x=df['Cadena'], y=df['Pct_Ex'],
        name='% Exceso', marker_color=PALETTE['accent2'],
        hovertemplate='<b>%{x}</b><br>%{y:.1f}%<extra></extra>'), row=1, col=2)
    fig.update_layout(**_layout('Salud de Inventario por Cadena'),
                      barmode='group', height=420,
                      legend=dict(orientation='h', y=-0.15))
    fig.update_yaxes(gridcolor=PALETTE['grid'])
    return fig


# ── Precio por Marca ──────────────────────────────────────────────────────────

def grafica_precio_marca(base: pd.DataFrame) -> go.Figure:
    precio = pd.to_numeric(base.get('Precio_promedio', pd.Series([])), errors='coerce')
    df = base.copy()
    df['_precio'] = precio
    df = df[df['_precio'] > 0].groupby('MARCA').agg(
        Prom=('_precio', 'mean'),
        Max=('_precio', 'max'),
        Min=('_precio', 'min'),
    ).reset_index().sort_values('Prom', ascending=False)

    colors = px.colors.qualitative.Set2
    fig = go.Figure()
    for i, row in df.iterrows():
        fig.add_trace(go.Bar(
            x=[row['MARCA']], y=[row['Prom']],
            name=row['MARCA'],
            marker_color=colors[i % len(colors)],
            hovertemplate=f'<b>{row["MARCA"]}</b><br>Prom: ${row["Prom"]:.2f}<br>Max: ${row["Max"]:.2f}<br>Min: ${row["Min"]:.2f}<extra></extra>',
        ))
    layout = _layout('Precio Promedio por Marca')
    layout['yaxis']['tickprefix'] = '$'
    layout['showlegend'] = False
    fig.update_layout(**layout)
    return fig


# ── Tendencia por Vendedor ────────────────────────────────────────────────────

def grafica_tendencia_vendedor(base: pd.DataFrame) -> go.Figure:
    if 'Vendedores' not in base.columns:
        return go.Figure()
    df = (
        base.groupby(['SEMANA', 'Vendedores'])['Venta Sell Out Importe Venta']
        .sum().reset_index().sort_values('SEMANA')
    )
    colores_v = [PALETTE['accent'], PALETTE['primary'], PALETTE['accent2'], PALETTE['neutral']]
    fig = go.Figure()
    for i, v in enumerate(df['Vendedores'].unique()):
        d = df[df['Vendedores'] == v]
        fig.add_trace(go.Scatter(
            x=d['SEMANA'], y=d['Venta Sell Out Importe Venta'],
            name=v, mode='lines+markers',
            line=dict(color=colores_v[i % len(colores_v)], width=2),
            hovertemplate=f'<b>{v}</b> S%{{x}}: $%{{y:,.0f}}<extra></extra>',
        ))
    layout = _layout('Tendencia Semanal por Vendedor — 2026')
    layout['yaxis']['tickprefix'] = '$'
    layout['xaxis']['title'] = 'Semana'
    layout['legend'] = dict(orientation='h', y=1.1)
    fig.update_layout(**layout)
    return fig
