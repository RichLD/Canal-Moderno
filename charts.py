import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

PALETTE = {
    'primary': '#1B2A4A',
    'accent': '#E8501A',
    'accent2': '#F5A623',
    'positive': '#27AE60',
    'negative': '#E74C3C',
    'neutral': '#95A5A6',
    'bg': '#F8F9FA',
    'grid': '#E9ECEF',
}


def _layout_base(title: str) -> dict:
    return dict(
        title=dict(text=f'<b>{title}</b>', font=dict(size=16, color=PALETTE['primary'])),
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(family='Arial', color=PALETTE['primary']),
        margin=dict(l=40, r=20, t=60, b=40),
        xaxis=dict(gridcolor=PALETTE['grid'], linecolor=PALETTE['grid']),
        yaxis=dict(gridcolor=PALETTE['grid'], linecolor=PALETTE['grid']),
        hovermode='x unified',
    )


# ─── 1. YoY Ventas ───────────────────────────────────────────────────────────

def grafica_yoy(base: pd.DataFrame) -> go.Figure:
    """Comparativo de ventas semanales año a año."""
    df = base.copy()
    años = sorted(df['AÑO'].unique())

    fig = go.Figure()
    colores = [PALETTE['neutral'], PALETTE['primary']]

    for i, año in enumerate(años[-2:]):          # solo últimos 2 años
        df_año = (
            df[df['AÑO'] == año]
            .groupby('SEMANA')['Venta Sell Out Importe Venta']
            .sum()
            .reset_index()
            .sort_values('SEMANA')
        )

        es_actual = (i == len(años[-2:]) - 1)
        color = PALETTE['accent'] if es_actual else PALETTE['neutral']
        dash = 'solid' if es_actual else 'dot'
        width = 3 if es_actual else 1.5

        fig.add_trace(go.Scatter(
            x=df_año['SEMANA'],
            y=df_año['Venta Sell Out Importe Venta'],
            name=str(año),
            mode='lines+markers',
            line=dict(color=color, dash=dash, width=width),
            marker=dict(size=4),
            hovertemplate=f'<b>{año}</b><br>Semana: %{{x}}<br>Venta: $%{{y:,.0f}}<extra></extra>',
        ))

    layout = _layout_base('Venta Sell Out — Comparativo YoY')
    layout['yaxis']['tickprefix'] = '$'
    layout['xaxis']['title'] = 'Semana'
    layout['yaxis']['title'] = 'Importe Venta'
    layout['legend'] = dict(orientation='h', y=1.1)
    fig.update_layout(**layout)
    return fig


# ─── 2. Ventas por Cadena ─────────────────────────────────────────────────────

def grafica_ventas_cadena(base: pd.DataFrame) -> go.Figure:
    """Bar chart horizontal de ventas totales por cadena."""
    df = (
        base.groupby('Cadena')['Venta Sell Out Importe Venta']
        .sum()
        .sort_values()
        .reset_index()
    )
    total = df['Venta Sell Out Importe Venta'].sum()
    df['Participacion'] = df['Venta Sell Out Importe Venta'] / total * 100

    fig = go.Figure(go.Bar(
        x=df['Venta Sell Out Importe Venta'],
        y=df['Cadena'],
        orientation='h',
        marker_color=PALETTE['primary'],
        text=df['Participacion'].map(lambda x: f'{x:.1f}%'),
        textposition='outside',
        hovertemplate='<b>%{y}</b><br>Venta: $%{x:,.0f}<extra></extra>',
    ))
    layout = _layout_base('Ventas por Cadena')
    layout['xaxis']['tickprefix'] = '$'
    layout['xaxis']['title'] = 'Importe Total'
    layout.pop('hovermode')
    fig.update_layout(**layout)
    return fig


# ─── 3. ABC ──────────────────────────────────────────────────────────────────

def grafica_abc(df_abc: pd.DataFrame) -> go.Figure:
    """Pareto ABC: barras por UPC + línea acumulada."""
    df = df_abc.dropna(subset=['Categoria_ABC']).copy()
    df = df.sort_values('Venta Sell Out Importe Venta', ascending=False).head(40)

    color_map = {'A': PALETTE['accent'], 'B': PALETTE['accent2'], 'C': PALETTE['neutral']}
    colores = df['Categoria_ABC'].astype(str).map(color_map)

    fig = make_subplots(specs=[[{'secondary_y': True}]])
    fig.add_trace(
        go.Bar(
            x=df['Descripcion'].astype(str).str[:25],
            y=df['Venta Sell Out Importe Venta'],
            marker_color=colores,
            name='Venta',
            hovertemplate='<b>%{x}</b><br>$%{y:,.0f}<extra></extra>',
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=df['Descripcion'].astype(str).str[:25],
            y=df['Perc_Acum'] * 100,
            mode='lines+markers',
            line=dict(color=PALETTE['primary'], width=2),
            name='% Acum.',
            hovertemplate='%{y:.1f}%<extra></extra>',
        ),
        secondary_y=True,
    )
    fig.update_layout(**_layout_base('Clasificación ABC — Top Productos'))
    fig.update_yaxes(title_text='Importe Venta', tickprefix='$', secondary_y=False, gridcolor=PALETTE['grid'])
    fig.update_yaxes(title_text='% Acumulado', ticksuffix='%', secondary_y=True, gridcolor=PALETTE['grid'])
    fig.update_xaxes(tickangle=-45, gridcolor=PALETTE['grid'])
    return fig


# ─── 4. Inventario por Cadena ─────────────────────────────────────────────────

def grafica_inventario(base: pd.DataFrame) -> go.Figure:
    """Días de inventario promedio y % agotados por cadena."""
    df = base.groupby('Cadena').agg(
        Dias_Inv_Prom=('Dias_Inventario', 'mean'),
        Pct_Agotado=('Es_Agotado', 'mean'),
        Pct_Exceso=('Exceso_Stock', 'mean'),
    ).reset_index()
    df['Pct_Agotado'] *= 100
    df['Pct_Exceso'] *= 100
    df = df.sort_values('Dias_Inv_Prom', ascending=False)

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Días de Cobertura Promedio', '% Agotados vs % Exceso de Stock'),
    )

    fig.add_trace(go.Bar(
        x=df['Cadena'], y=df['Dias_Inv_Prom'],
        marker_color=PALETTE['primary'],
        name='Días cobertura',
        hovertemplate='<b>%{x}</b><br>%{y:.1f} días<extra></extra>',
    ), row=1, col=1)

    # Línea de referencia 14 días
    fig.add_hline(y=14, line_dash='dash', line_color=PALETTE['accent'],
                  annotation_text='14 días', row=1, col=1)

    fig.add_trace(go.Bar(
        x=df['Cadena'], y=df['Pct_Agotado'],
        name='% Agotados', marker_color=PALETTE['negative'],
        hovertemplate='<b>%{x}</b><br>Agotado: %{y:.1f}%<extra></extra>',
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=df['Cadena'], y=df['Pct_Exceso'],
        name='% Exceso', marker_color=PALETTE['accent2'],
        hovertemplate='<b>%{x}</b><br>Exceso: %{y:.1f}%<extra></extra>',
    ), row=1, col=2)

    fig.update_layout(
        **_layout_base('Salud de Inventario por Cadena'),
        barmode='group',
        legend=dict(orientation='h', y=-0.15),
        height=420,
    )
    fig.update_yaxes(gridcolor=PALETTE['grid'])
    return fig


# ─── 5. Precio promedio por marca ─────────────────────────────────────────────

def grafica_precio_marca(base: pd.DataFrame) -> go.Figure:
    """Box plot de precio promedio por marca (sin ceros)."""
    df = base[base['Precio_promedio'] > 0].copy()

    marcas_orden = (
        df.groupby('MARCA')['Precio_promedio']
        .median()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig = go.Figure()
    colors = px.colors.qualitative.Set2
    for i, marca in enumerate(marcas_orden):
        datos = df[df['MARCA'] == marca]['Precio_promedio']
        fig.add_trace(go.Box(
            y=datos,
            name=marca,
            marker_color=colors[i % len(colors)],
            boxmean=True,
        ))

    layout = _layout_base('Distribución de Precio Promedio por Marca')
    layout['yaxis']['tickprefix'] = '$'
    layout['yaxis']['title'] = 'Precio promedio (MXN)'
    layout['showlegend'] = False
    fig.update_layout(**layout)
    return fig


# ─── 6. Tendencia semanal por vendedor ───────────────────────────────────────

def grafica_tendencia_vendedor(base: pd.DataFrame) -> go.Figure:
    """Línea de venta acumulada semanal por vendedor, año más reciente."""
    año_max = base['AÑO'].max()
    df = (
        base[base['AÑO'] == año_max]
        .groupby(['SEMANA', 'Vendedores'])['Venta Sell Out Importe Venta']
        .sum()
        .reset_index()
        .sort_values('SEMANA')
    )

    vendedores = df['Vendedores'].unique()
    colores_v = [PALETTE['accent'], PALETTE['primary'], PALETTE['accent2'], PALETTE['neutral']]

    fig = go.Figure()
    for i, v in enumerate(vendedores):
        d = df[df['Vendedores'] == v]
        fig.add_trace(go.Scatter(
            x=d['SEMANA'], y=d['Venta Sell Out Importe Venta'],
            name=v, mode='lines+markers',
            line=dict(color=colores_v[i % len(colores_v)], width=2),
            hovertemplate=f'<b>{v}</b><br>S%{{x}}: $%{{y:,.0f}}<extra></extra>',
        ))

    layout = _layout_base(f'Tendencia Semanal por Vendedor — {año_max}')
    layout['yaxis']['tickprefix'] = '$'
    layout['xaxis']['title'] = 'Semana'
    layout['legend'] = dict(orientation='h', y=1.1)
    fig.update_layout(**layout)
    return fig


# ─── 7. KPIs resumen ─────────────────────────────────────────────────────────

def calcular_kpis(base: pd.DataFrame) -> dict:
    año_max = base['AÑO'].max()
    año_ant = año_max - 1

    venta_actual = base[base['AÑO'] == año_max]['Venta Sell Out Importe Venta'].sum()
    venta_anterior = base[base['AÑO'] == año_ant]['Venta Sell Out Importe Venta'].sum()
    var_yoy = ((venta_actual / venta_anterior) - 1) * 100 if venta_anterior > 0 else None

    pct_agotado = base['Es_Agotado'].mean() * 100
    dias_inv = base[base['Dias_Inventario'] > 0]['Dias_Inventario'].mean()
    sell_through = base['Sell_Through_Rate'].dropna().mean() * 100
    tiendas = base['NomTienda'].nunique()
    skus = base['UPC'].nunique()

    return {
        'venta_actual': venta_actual,
        'var_yoy': var_yoy,
        'pct_agotado': pct_agotado,
        'dias_inventario': dias_inv,
        'sell_through': sell_through,
        'tiendas': tiendas,
        'skus': skus,
        'año_max': año_max,
    }
