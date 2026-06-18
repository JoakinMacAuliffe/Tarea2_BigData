import numpy as np
from collections import Counter
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD


def tokenizar(texto: str) -> list[str]:
    """Convierte texto en lista de tokens en minúsculas."""
    return re.findall(r'\b[a-z0-9]+\b', texto.lower())


def construir_vocabulario(corpus: list[str]) -> list[str]:
    """Retorna vocabulario ordenado del corpus."""
    vocab = set()
    for doc in corpus:
        vocab.update(tokenizar(doc))
    return sorted(vocab)


def vectorizar_tf(doc: str, vocab: list[str]) -> np.ndarray:
    """Convierte un documento en vector de frecuencias."""
    conteo = Counter(tokenizar(doc))
    return np.array([conteo.get(w, 0) for w in vocab], dtype=float)


def similitud_coseno(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Calcula la similitud de coseno entre dos vectores."""
    norma_a = np.linalg.norm(vec_a)
    norma_b = np.linalg.norm(vec_b)
    if norma_a == 0 or norma_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norma_a * norma_b))


def cargar_bbc_news(ruta_csv: str) -> pd.DataFrame:
    """Carga el dataset BBC News y crea una columna de texto unificada."""
    df = pd.read_csv(ruta_csv, sep='\t')
    df['title'] = df['title'].fillna('').astype(str).str.strip()
    df['content'] = df['content'].fillna('').astype(str).str.strip()
    df['texto'] = (df['title'] + ' ' + df['content']).str.strip()
    df = df[df['texto'].astype(bool)].copy()
    return df


ruta_dataset = 'bbc-news-data.csv'
df_noticias = cargar_bbc_news(ruta_dataset)
documentos = df_noticias['texto'].tolist()
categorias = df_noticias['category'].tolist()

print('=' * 60)
print('BBC NEWS DATASET')
print('=' * 60)
print(f'Registros cargados: {len(df_noticias)}')
print(f'Categorías: {", ".join(sorted(df_noticias["category"].unique()))}')

# Corpus de ejemplo tomado del dataset
corpus = documentos[:5]

consulta = 'economy markets profits'

# Construcción del espacio vectorial
vocab = construir_vocabulario(corpus + [consulta])
vectores = [vectorizar_tf(doc, vocab) for doc in corpus]
vec_consulta = vectorizar_tf(consulta, vocab)

# Cálculo y ranking
similitudes = [similitud_coseno(vec_consulta, v) for v in vectores]
ranking = sorted(enumerate(similitudes), key=lambda x: x[1], reverse=True)

print("=" * 60)
print(f"Consulta: '{consulta}'")
print("=" * 60)
print(f"{'Rank':<5} {'Sim':>6}  Documento")
print("-" * 60)
for rank, (idx, sim) in enumerate(ranking, 1):
    print(f"{rank:<5} {sim:.4f}  {corpus[idx]}")

# Consultas diseñadas para el corpus BBC
consultas = [
    'business market profits',
    'sport football championship',
    'health medicine vaccine',
    'technology search engine artificial intelligence',
]

# Vectorización TF-IDF
vectorizador = TfidfVectorizer(
    ngram_range=(1, 2),          # unigramas y bigramas
    min_df=1,
    sublinear_tf=True,           # log(1 + tf) en lugar de tf
    strip_accents='unicode',
)

matriz_tfidf = vectorizador.fit_transform(documentos)
vectores_consultas = vectorizador.transform(consultas)

# Búsqueda y ranking
similitudes = cosine_similarity(vectores_consultas, matriz_tfidf)

print("\n" + "=" * 70)
print("RESULTADOS DE BÚSQUEDA CON TF-IDF + SIMILITUD DE COSENO")
print("=" * 70)

for i, consulta in enumerate(consultas):
    scores = similitudes[i]
    ranking_idx = scores.argsort()[::-1][:3]   # Top-3 documentos
    print(f"\nConsulta: '{consulta}'")
    print("-" * 70)
    for pos, idx in enumerate(ranking_idx, 1):
        print(f"  {pos}. [{scores[idx]:.4f}] ({categorias[idx]}) {df_noticias.iloc[idx]['title']}")

# Matriz de similitud entre documentos
sample_size = min(25, len(df_noticias))
df_muestra = df_noticias.sample(n=sample_size, random_state=42).reset_index(drop=True)
documentos_muestra = df_muestra['texto'].tolist()
matriz_tfidf_muestra = vectorizador.transform(documentos_muestra)
sim_docs = cosine_similarity(matriz_tfidf_muestra)
df_sim = pd.DataFrame(
    sim_docs,
    index=[f"D{i+1}" for i in range(len(df_muestra))],
    columns=[f"D{i+1}" for i in range(len(df_muestra))],
)

print("\n\nMATRIZ DE SIMILITUD ENTRE DOCUMENTOS (primeros 5):")
print(df_sim.iloc[:5, :5].round(3).to_string())

# Reducción a 2D con TruncatedSVD, que trabaja directamente sobre matrices dispersas
svd = TruncatedSVD(n_components=2, random_state=42)
coords = svd.fit_transform(matriz_tfidf_muestra)

# Etiquetas abreviadas
etiquetas = [f"D{i+1}: {n[:25]}..." for i, n in enumerate(documentos_muestra)]

# Gráfico
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter plot de documentos
ax1 = axes[0]
scatter = ax1.scatter(coords[:, 0], coords[:, 1], c='steelblue',
                      s=120, zorder=5, edgecolors='white', linewidths=1.5)
for i, (x, y) in enumerate(coords):
    ax1.annotate(f"D{i+1} ({df_muestra.loc[i, 'category']})", (x, y), textcoords="offset points",
                 xytext=(6, 6), fontsize=9, color='#333333')
ax1.set_title("Documentos BBC en espacio 2D (TruncatedSVD sobre TF-IDF)", fontsize=11)
ax1.set_xlabel("Componente Principal 1")
ax1.set_ylabel("Componente Principal 2")
ax1.grid(True, alpha=0.3)

# Heatmap de similitud
ax2 = axes[1]
n = len(df_muestra)
labels_cortas = [f"D{i+1}" for i in range(n)]
im = ax2.imshow(sim_docs, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels(labels_cortas, fontsize=8)
ax2.set_yticklabels(labels_cortas, fontsize=8)
for i in range(n):
    for j in range(n):
        ax2.text(j, i, f"{sim_docs[i, j]:.2f}",
                 ha='center', va='center', fontsize=6,
                 color='black' if sim_docs[i, j] < 0.6 else 'white')
plt.colorbar(im, ax=ax2, label='Similitud de Coseno')
ax2.set_title("Matriz de similitud entre una muestra del BBC", fontsize=11)

plt.tight_layout()
plt.show()