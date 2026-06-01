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


def _cols_a_str(base: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte todas las columnas category/object/ArrowDtype a str nativo de Python.
    Necesario porque pandas 3+ con pyarrow usa ArrowDtype internamente para parquet,
    lo que rompe .astype(int) y la concatenación con +.
    """
    for col in base.columns:
        if hasattr(base[col], 'cat') or str(base[col].dtype) in ('object', 'string'):
            base[col] = base[col].astype(object).fillna('').astype(str)
        # ArrowDtype aparece como string[pyarrow] o large_string[pyarrow]
        dtype_str = str(base[col].dtype)
        if 'string' in dtype_str or 'large_string' in dtype_str:
            base[col] = base[col].astype(object).fillna('').astype(str)
    return base


def parsear_tiempo(base: pd.DataFrame) -> pd.DataFrame:
    """Extrae AÑO, SEMANA y Fecha usando aritmética entera — evita problemas de Arrow strings."""
    base = base.copy()
    # pd.to_numeric maneja int32, float, str, Arrow → siempre devuelve numpy int
    tiempo = pd.to_numeric(base['Tiempo'], errors='coerce').fillna(0).astype(int)
    base['AÑO'] = (tiempo // 100)
    base['SEMANA'] = (tiempo % 100)
    # Eliminar filas con Tiempo=0 (NaN originales)
    base = base[base['AÑO'] > 0].copy()
    base['Fecha'] = pd.to_datetime(
        base['AÑO'].astype(str) + '-' + base['SEMANA'].astype(str) + '-1',
        format='%Y-%W-%w',
        errors='coerce'
    )
    return base


def limpiar_base(base: pd.DataFrame) -> pd.DataFrame:
    """Limpieza general y creación de columnas derivadas."""
    base = base.copy()

    # Normalizar columnas con espacio extra
    base.columns = base.columns.str.strip()

    # Convertir todas las columnas de texto a str nativo (fix Arrow / category)
    base = _cols_a_str(base)

    # OH negativo → 0
    base['OH Piezas'] = pd.to_numeric(base['OH Piezas'], errors='coerce').fillna(0)
    base['OH Piezas'] = np.where(base['OH Piezas'] < 0, 0, base['OH Piezas'])

    # Vendedores
    base['Vendedores'] = base['Cadena'].map(VENDEDORES_MAP).fillna('Sin Vendedor Asignado')

    # Cluster — seguro porque ya son str nativos
    base['Cluster'] = (
        base['Cadena'] + '_' +
        base['Formato'].fillna('') + '_' +
        base['Estado'].fillna('')
    )

    return base


def calcular_metricas_precio(base: pd.DataFrame) -> pd.DataFrame:
    """Precio promedio y precio con IEPS."""
    base = base.copy()
    ventas_imp = pd.to_numeric(base['Venta Sell Out Importe Venta'], errors='coerce').fillna(0)
    ventas_pzs = pd.to_numeric(base['Venta Sell Out Piezas'], errors='coerce').fillna(0)
    precio = ventas_imp / ventas_pzs
    base['Precio_promedio'] = precio.replace([np.inf, -np.inf], np.nan).fillna(0)
    base['Precio_Promedio_IEPS'] = base['Precio_promedio'] * 1.08
    return base


def calcular_metricas_inventario(base: pd.DataFrame) -> pd.DataFrame:
    """Métricas de inventario corregidas."""
    base = base.copy()
    ventas = pd.to_numeric(base['Venta Sell Out Piezas'], errors='coerce').fillna(0)
    oh = pd.to_numeric(base['OH Piezas'], errors='coerce').fillna(0)

    base['Dias_Inventario'] = np.where(ventas > 0, (oh / ventas) * 7, 0)

    base['Rotacion_Inventario'] = np.where(oh > 0, ventas / oh, np.nan)
    base['Rotacion_Inventario'] = base['Rotacion_Inventario'].replace([np.inf, -np.inf], np.nan)

    denominador = ventas + oh
    base['Sell_Through_Rate'] = np.where(denominador > 0, ventas / denominador, np.nan)

    base['Es_Agotado'] = (oh <= 0).astype(int)
    base['Exceso_Stock'] = (base['Dias_Inventario'] > 42).astype(int)

    return base


def calcular_abc(base: pd.DataFrame) -> pd.DataFrame:
    """Clasificación ABC por UPC según importe de venta acumulado."""
    ventas_col = pd.to_numeric(base['Venta Sell Out Importe Venta'], errors='coerce').fillna(0)
    base = base.copy()
    base['_venta_num'] = ventas_col

    df_abc = (
        base.groupby('UPC')['_venta_num']
        .sum()
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={'_venta_num': 'Venta Sell Out Importe Venta'})
    )
    total = df_abc['Venta Sell Out Importe Venta'].sum()
    df_abc['CumSum'] = df_abc['Venta Sell Out Importe Venta'].cumsum()
    df_abc['Perc_Acum'] = df_abc['CumSum'] / total if total > 0 else 0
    df_abc['Categoria_ABC'] = pd.cut(
        df_abc['Perc_Acum'],
        bins=[0, 0.80, 0.95, 1.0],
        labels=['A', 'B', 'C']
    )
    desc = base.groupby('UPC')['Descripcion'].first().reset_index()
    df_abc = desc.merge(df_abc, on='UPC', how='left')
    return df_abc


def calcular_mix_cluster(base: pd.DataFrame) -> pd.DataFrame:
    """Participación de cada UPC dentro de su cluster."""
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
    return mix.sort_values(['Cluster', 'Participacion_Mix'], ascending=[True, False])


def calcular_venta_promedio_tienda(base: pd.DataFrame) -> pd.DataFrame:
    """Promedio de venta por tienda ponderado por semanas activas."""
    semanas_activas = (
        base[pd.to_numeric(base['Venta Sell Out Piezas'], errors='coerce').fillna(0) > 0]
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
    """Pipeline completo de procesamiento."""
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
