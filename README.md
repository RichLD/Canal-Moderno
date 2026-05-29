# Dashboard Sell Out — Arcor México

Dashboard interactivo de ventas Sell Out construido con Streamlit.

## Estructura del proyecto

```
arcor_dashboard/
├── app.py                        ← Dashboard principal
├── requirements.txt              ← Dependencias
└── utils/
    ├── feature_engineering.py   ← Cálculo de métricas y KPIs
    └── charts.py                ← Todas las gráficas (Plotly)
```

## Cómo usar

1. El usuario carga el CSV desde la interfaz (sidebar)
2. El dashboard procesa y cachea los datos automáticamente
3. Los filtros (año, cadena, vendedor, marca, estado) actualizan todas las visualizaciones

## Métricas incluidas

| Métrica | Fórmula |
|---|---|
| Días de cobertura | `(OH / Venta_semanal) × 7` |
| Rotación inventario | `Ventas / OH` |
| Sell Through Rate | `Ventas / (Ventas + OH)` |
| Precio promedio | `Importe / Piezas` |
| Precio con IEPS | `Precio × 1.08` |
| Clasificación ABC | Pareto acumulado (A=80%, B=15%, C=5%) |

## Deploy en Streamlit Community Cloud

1. Sube este repositorio a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu cuenta de GitHub
4. Selecciona el repo, la rama `main` y el archivo `app.py`
5. Click en **Deploy**

El CSV **no se guarda en el servidor** — cada usuario lo carga localmente en su sesión.

## Columnas requeridas en el CSV

| Columna | Tipo |
|---|---|
| Cadena | texto |
| UPC | numérico |
| Descripcion | texto |
| MARCA (o MARCA ) | texto |
| CATEGORIA | texto |
| Formato | texto |
| NumTienda | texto |
| NomTienda | texto |
| Ciudad | texto |
| Estado | texto |
| Tiempo | entero (ej. 202510) |
| Venta Sell Out Piezas | float |
| Venta Sell Out Importe Venta | float |
| OH Piezas | float |
