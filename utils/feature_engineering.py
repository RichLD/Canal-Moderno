import pandas as pd
import numpy as np


VENDEDORES_MAP = {
    'Casa_Ley': 'JOSAFAT',
    'Chedraui': 'ORLANDO',
    'City': 'JOSAFAT',
    'City_Fresko': 'ORLANDO',
    'Grupo_Control': 'ORLANDO',
    'HEB': 'JOSAFAT',
    'Sams_Madrid': 'ORLANDO',
    'Soriana': 'JOSAFAT',
    'Wal-Mart': 'ORLANDO',
    'Wal-Mart - CHECK OUT': 'ORLANDO',
}


def parsear_tiempo(base: pd.DataFrame) -> pd.DataFrame:
    """Extrae AÑO, SEMANA y Fecha a partir de la columna Tiempo (ej. 202510)."""
    base = base.copy()
    # Convertir a int nativo — funciona con int32, float, Arrow strings
    tiempo = pd.to_numeric(base['Tiempo'], errors='coerce').fillna(0).astype(int)
    base['AÑO'] = (tiempo // 100)
    base['SEMANA'] = (tiempo % 100)
    base['Fecha'] = pd.to_datetime(
        base['AÑO'].astype(str) + '-' + base['SEMANA'].astype(str) + '-1',
        format='%Y-%W-%w',
        errors='coerce'
    )
    return base


def limpiar_base(base: pd.DataFrame) -> pd.DataFrame:
    """Limpieza general: negativos en OH, columna MARCA con posible espacio."""
    base = base.copy()

    # Normalizar nombre de columna MARCA (puede venir con espacio)
    if 'MARCA ' in base.columns and 'MARCA' not in base.columns:
        base = base.rename(columns={'MARCA ': 'MARCA'})

    # OH negativo → 0
    base['OH Piezas'] = np.where(base['OH Piezas'] < 0, 0, base['OH Piezas'])

    # Vendedores
    base['Vendedores'] = base['Cadena'].map(VENDEDORES_MAP).fillna('Sin Vendedor Asignado')

    # Cluster
    base['Cluster'] = base['Cadena'] + '_' + base['Formato'].fillna('') + '_' + base['Estado'].fillna('')

    return base


def calcular_metricas_precio(base: pd.DataFrame) -> pd.DataFrame:
    """Precio promedio y precio con IEPS."""
    base = base.copy()
    precio = base['Venta Sell Out Importe Venta'] / base['Venta Sell Out Piezas']
    base['Precio_promedio'] = precio.replace([np.inf, -np.inf], np.nan).fillna(0)
    base['Precio_Promedio_IEPS'] = base['Precio_promedio'] * 1.08
    return base


def calcular_metricas_inventario(base: pd.DataFrame) -> pd.DataFrame:
    """
    Métricas de inventario corregidas:
    - Días inventario: (OH / Venta_semanal) * 7  → cobertura real en días
    - Rotación: Ventas / OH  (cuántas veces rotas el stock; convención estándar)
    - Sell Through: Ventas / (Ventas + OH)  → proxy aceptable sin OH inicial
    - Banderas: agotado y exceso de stock (>42 días = 6 semanas)
    """
    base = base.copy()
    ventas = base['Venta Sell Out Piezas']
    oh = base['OH Piezas']

    # Días de cobertura (base semanal → *7)
    base['Dias_Inventario'] = np.where(
        ventas > 0,
        (oh / ventas) * 7,
        0
    )

    # Rotación: ventas / OH (veces que se rota el inventario)
    base['Rotacion_Inventario'] = np.where(
        oh > 0,
        ventas / oh,
        np.nan
    )
    base['Rotacion_Inventario'] = base['Rotacion_Inventario'].replace([np.inf, -np.inf], np.nan)

    # Sell Through Rate (proxy sin OH inicial)
    denominador = ventas + oh
    base['Sell_Through_Rate'] = np.where(
        denominador > 0,
        ventas / denominador,
        np.nan
    )

    # Banderas
    base['Es_Agotado'] = (oh <= 0).astype(int)
    base['Exceso_Stock'] = (base['Dias_Inventario'] > 42).astype(int)   # >6 semanas

    return base


def calcular_abc(base: pd.DataFrame) -> pd.DataFrame:
    """Clasificación ABC por UPC según importe de venta acumulado."""
    df_abc = (
        base.groupby('UPC')['Venta Sell Out Importe Venta']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    df_abc['CumSum'] = df_abc['Venta Sell Out Importe Venta'].cumsum()
    total = df_abc['Venta Sell Out Importe Venta'].sum()
    df_abc['Perc_Acum'] = df_abc['CumSum'] / total if total > 0 else 0
    df_abc['Categoria_ABC'] = pd.cut(
        df_abc['Perc_Acum'],
        bins=[0, 0.80, 0.95, 1.0],
        labels=['A', 'B', 'C']
    )

    # Traer descripción
    desc = base.groupby('UPC')['Descripcion'].first().reset_index()
    df_abc = desc.merge(df_abc, on='UPC', how='left')
    return df_abc


def calcular_mix_cluster(base: pd.DataFrame) -> pd.DataFrame:
    """Participación de cada UPC dentro de su cluster (Cadena+Formato+Estado)."""
    mix = (
        base.groupby(['Cluster', 'UPC'])
        .agg({'Venta Sell Out Piezas': 'sum', 'Venta Sell Out Importe Venta': 'sum'})
        .reset_index()
    )
    total_cluster = mix.groupby('Cluster')['Venta Sell Out Importe Venta'].transform('sum')
    mix['Participacion_Mix'] = np.where(
        total_cluster > 0,
        mix['Venta Sell Out Importe Venta'] / total_cluster,
        0
    )
    mix = mix.sort_values(['Cluster', 'Participacion_Mix'], ascending=[True, False])
    return mix


def calcular_venta_promedio_tienda(base: pd.DataFrame) -> pd.DataFrame:
    """
    Promedio de venta por tienda ponderado por semanas activas.
    Evita el sesgo de tiendas con pocas semanas de actividad.
    """
    semanas_activas = (
        base[base['Venta Sell Out Piezas'] > 0]
        .groupby('NomTienda')['SEMANA']
        .nunique()
        .rename('Semanas_Activas')
    )
    venta_total_tienda = (
        base.groupby('NomTienda')['Venta Sell Out Piezas']
        .sum()
        .rename('Venta_Total')
    )
    resumen = pd.concat([venta_total_tienda, semanas_activas], axis=1).fillna(0)
    resumen['Venta_Prom_Semanal_Tienda'] = np.where(
        resumen['Semanas_Activas'] > 0,
        resumen['Venta_Total'] / resumen['Semanas_Activas'],
        0
    )
    return resumen.reset_index()


def procesar_completo(base: pd.DataFrame) -> dict:
    """
    Pipeline completo. Devuelve un diccionario con:
    - 'base': DataFrame enriquecido
    - 'abc': clasificación ABC
    - 'mix_cluster': participación por cluster
    - 'venta_tienda': resumen por tienda
    """
    base = parsear_tiempo(base)
    base = limpiar_base(base)
    base = calcular_metricas_precio(base)
    base = calcular_metricas_inventario(base)

    abc = calcular_abc(base)
    mix_cluster = calcular_mix_cluster(base)
    venta_tienda = calcular_venta_promedio_tienda(base)

    return {
        'base': base,
        'abc': abc,
        'mix_cluster': mix_cluster,
        'venta_tienda': venta_tienda,
    }
