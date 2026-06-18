# Tarea 2 Big Data

Script de analisis de noticias BBC usando:

- TF manual para una consulta de ejemplo
- TF-IDF con similitud de coseno
- Matriz de similitud entre documentos
- Reduccion dimensional con `TruncatedSVD`
- Visualizacion con `matplotlib`

## Requisitos

- Python 3.10 o superior
- El archivo `bbc-news-data.csv` ya viene incluido en el repositorio

## Crear y activar un entorno virtual

En Windows PowerShell, desde la carpeta del proyecto:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Si usas CMD en lugar de PowerShell:

```cmd
py -3 -m venv .venv
.venv\Scripts\activate.bat
```

## Instalar dependencias

Con el entorno virtual activado:

```powershell
pip install -r requirements.txt
```

## Ejecutar el script

```powershell
python main.py
```

## Que hace el script

- Carga el dataset `bbc-news-data.csv`
- Combina `title` y `content` en una sola columna de texto
- Calcula similitud entre documentos con TF-IDF
- Muestra resultados de busqueda para varias consultas
- Genera una visualizacion 2D con `TruncatedSVD`

## Estructura esperada

```text
Tarea2_BigData/
├─ main.py
├─ requirements.txt
├─ bbc-news-data.csv
└─ README.md
```

## Notas

- La ventana de la figura se abre al final de la ejecucion con `plt.show()`.
- No hace falta descargar el CSV aparte; debe estar en la raiz del repositorio junto a `main.py`.
- Si ejecutas desde PowerShell y la activacion del entorno virtual falla por politica de ejecucion, puedes usar:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
