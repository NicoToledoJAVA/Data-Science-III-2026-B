# Módulo 3: Clasificador Supervisado con TF-IDF

## Descripción
Este proyecto implementa un pipeline de clasificación de texto utilizando técnicas clásicas de Procesamiento de Lenguaje Natural (NLP). El objetivo es clasificar noticias de AG News en 4 categorías utilizando vectorización TF-IDF y modelos de Machine Learning clásicos.

## Dataset
- **AG News**: 10,000 artículos de noticias en inglés
- **Categorías**: World (1), Sports (2), Business (3), Sci/Tech (4)
- **Split**: 8,000 entrenamiento / 2,000 prueba (proporcionados separadamente)

## Estructura del Proyecto

```
.
├── clasificador_tfidf.py          # Script principal del módulo 3
├── cargar_dataset.py              # Cargador de datasets (módulo 2)
├── pipeline_eda.py               # Pipeline de preprocesamiento (módulo 2)
├── requirements.txt              # Dependencias del proyecto
├── salidas/                      # Outputs del módulo 2 (preprocesamiento)
├── salidas_mod3/                 # Outputs del módulo 3 (clasificación)
│   ├── *.pkl                     # Modelos y vectorizador serializados
│   ├── *.png                     # Gráficos y matrices de confusión
│   ├── *.json                    # Resultados en formato JSON
│   ├── *.txt                     # Reportes de clasificación
│   └── *.csv                     # Análisis y comparaciones
└── ag_news/                      # Datasets originales
    ├── ag_news_train.csv
    └── ag_news_test.csv
```

## Pipeline Implementado

### 1. Preprocesamiento (Reutilizado del Módulo 2)
- Limpieza de HTML, URLs, emails y entidades
- Normalización a minúsculas
- Lematización con spaCy
- Filtrado de stop-words (preservando negaciones)
- Tokenización

### 2. Vectorización TF-IDF
- **Separación estricta**: `fit_transform()` solo en train, `transform()` en test
- **Parámetros experimentados**:
  - `max_features`: 5k, 10k, 20k
  - `ngram_range`: (1,1) unigramas, (1,2) unigramas+bigramas
  - `min_df=2`: Ignora términos en menos de 2 documentos
  - `max_df=0.95`: Ignora términos en más del 95% de documentos
  - `stop_words='english'`: Filtra stop-words en inglés

### 3. Modelos Baseline
1. **Naive Bayes Multinomial**: Modelo probabilístico rápido y eficiente para texto
2. **SVM (LinearSVC)**: Máquinas de vectores de soporte lineales
3. **Regresión Logística**: Modelo lineal con probabilidades

### 4. Evaluación
- **Accuracy**: Precisión general
- **F1-Score (macro)**: Promedio armónico de precisión y recall
- **Matriz de confusión**: Visualización de errores por categoría
- **Cross-validation**: Validación de 5 folds en entrenamiento
- **Reporte por categoría**: Precisión, recall y F1-score individual

## Resultados Obtenidos

### Mejor Configuración TF-IDF
- **Configuración**: `bigram_20k` (unigramas + bigramas, 20k features)
- **Vocabulary size**: 20,000 términos
- **Selección**: por F1 Macro en validación cruzada (3 folds) sobre train — CV F1 Macro = 88.23%
- **Accuracy (Naive Bayes, test)**: 89.65%
- **F1 Macro (Naive Bayes, test)**: 89.67%

### Comparación de Modelos
| Modelo | Accuracy (test) | F1 Macro (test) | CV F1 Macro (train) |
|--------|-----------------|-----------------|---------------------|
| NaiveBayes | 89.65% | 89.67% | 89.12% ± 1.21% |
| SVM (LinearSVC) | 88.95% | 88.97% | 88.47% ± 1.17% |
| LogisticRegression | 89.25% | 89.26% | 88.56% ± 1.06% |

### Análisis por Categoría (Naive Bayes, test)
| Categoría | Precisión | Recall | F1-Score | Support |
|-----------|-----------|--------|----------|---------|
| Business | 83.05% | 87.20% | 85.07% | 500 |
| Sci/Tech | 88.91% | 86.60% | 87.74% | 500 |
| Sports | 95.45% | 96.40% | 95.92% | 500 |
| World | 91.51% | 88.40% | 89.93% | 500 |

**Categoría más difícil**: Business (F1-score: 85.07%)
**Categoría más fácil**: Sports (F1-score: 95.92%)

## Justificación de Decisiones

### Elección del Modelo (Naive Bayes)
- **Rendimiento**: Mejor F1 Macro en test (89.67%) y mejor CV F1 Macro (89.12%) entre los modelos probados
- **Eficiencia**: Entrena en segundos y predice rápido, ideal como baseline
- **Adecuación para texto**: Modelo probabilístico que funciona muy bien con matrices dispersas TF-IDF
- **Simplicidad**: Pocos hiperparámetros, menos riesgo de sobreajuste en alta dimensionalidad

> Nota: la configuración de TF-IDF se eligió con validación cruzada sobre train, no mirando el test. El test se usó una sola vez para la evaluación final de cada modelo.

### Parámetros TF-IDF
- **Bigramas (1,2)**: Capturan contexto local ("white house", "stock market") y mejoraron el F1 frente a unigramas solos
- **max_features=20k**: Balance entre cobertura de vocabulario y costo computacional/memoria
- **min_df=2**: Elimina términos raros que aparecen una sola vez y suelen ser ruido
- **max_df=0.95**: Elimina términos presentes en casi todos los documentos
- **stop_words='english'**: Refuerza el filtrado ya hecho en el preprocesamiento

### Prevención de Data Leakage
- **Separación estricta**: `fit_transform()` solo sobre train; `transform()` sobre test
- **Selección de hiperparámetros**: validación cruzada dentro de train (el vectorizador va dentro del Pipeline, así que cada fold ajusta su propio vocabulario)
- **Test intocable**: el set de test se usa únicamente para la evaluación final reportada

## Instalación y Ejecución

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Ejecutar pipeline completo
```bash
python clasificador_tfidf.py
```

### 3. Outputs generados
- **Modelos serializados**: `salidas_mod3/*.pkl`
- **Gráficos**: Matrices de confusión y análisis de features
- **Reportes**: Métricas detalladas en JSON y texto
- **Análisis**: Comparación de modelos y categorías en CSV

## Análisis de Errores

### Matriz de Confusión (Naive Bayes)
Filas = etiqueta real, columnas = predicción. Orden alfabético de clases: Business, Sci/Tech, Sports, World.

| Real \ Predicho | Business | Sci/Tech | Sports | World |
|-----------------|----------|----------|--------|-------|
| Business | **436** | 39 | 4 | 21 |
| Sci/Tech | 47 | **433** | 5 | 15 |
| Sports | 6 | 7 | **482** | 5 |
| World | 36 | 8 | 14 | **442** |

La matriz muestra que:
- **Business ↔ Sci/Tech**: confusión principal de Business (39 + 47 = 86 casos), por vocabulario compartido ("company", "technology", "market")
- **Business ↔ World**: segunda confusión relevante (21 + 36 = 57 casos), por noticias de economía internacional
- **Business**: peor recall (87.2%): pierde casos hacia Sci/Tech y World
- **Sports**: clase más separable (482/500 correctos)
- **World**: buena precisión (91.5%) pero recall 88.4%, con fugas hacia Business

### Features más Importantes (Regresión Logística)
Coeficientes positivos más altos por clase:
- **Business**: company, profit, oil, billion, economy, share, stock, price
- **Sci/Tech**: internet, software, scientist, microsoft, computer, nasa, technology, phone
- **Sports**: team, coach, win, game, season, cup, league, player, olympic
- **World**: iraq, president, leader, minister, nuclear, election, rebel, military, israel

## Mejoras Futuras

1. **Optimización de hiperparámetros**: GridSearch para TF-IDF y modelos
2. **Features adicionales**: Longitud de texto, presencia de números, etc.
3. **Ensamblado**: Voting classifier combinando múltiples modelos
4. **Word embeddings**: Uso de Word2Vec o FastText como alternativa
5. **Deep Learning**: Redes neuronales para comparación de performance

## Conclusiones

El pipeline implementado demuestra la efectividad de técnicas clásicas de NLP para clasificación de texto. Naive Bayes con TF-IDF (unigramas + bigramas, 20k features) alcanza 89.65% de accuracy y 89.67% de F1 Macro en test, mostrando que:

1. **TF-IDF es efectivo**: Captura la importancia relativa de términos sin necesidad de embeddings
2. **Bigramas mejoran el contexto**: La configuración (1,2) superó a los unigramas en la validación cruzada
3. **El preprocesamiento es crucial**: La limpieza y lematización del Módulo 2 se traducen directamente en tokens más informativos
4. **Sports es la clase más fácil**: Vocabulario muy distintivo y estable
5. **Business es la clase más difícil**: Su vocabulario se solapa con Sci/Tech (empresas tecnológicas) y World (economía internacional)

El proyecto cumple con todos los requisitos técnicos: separación estricta train/test, selección de hiperparámetros sin tocar el test, prevención de data leakage, evaluación rigurosa (Precision/Recall/F1 + matriz de confusión multiclase) y documentación completa.

## Archivos Entregables

| Archivo | Descripción |
|---------|-------------|
| `clasificador_tfidf.py` | Pipeline completo: carga, TF-IDF, entrenamiento y evaluación |
| `clasificador_tfidf.ipynb` | Notebook ejecutado con el mismo pipeline y gráficos inline |
| `README_MOD3.md` | Este documento con justificación y análisis |
| `requirements.txt` | Dependencias y versiones |
| `salidas_mod3/` | Modelos, reportes, matrices de confusión y gráficos |
| `salidas_mod3/pipeline_completo.pkl` | Pipeline entrenado listo para predecir texto crudo |

## Reproducibilidad

```bash
python -m pip install -r requirements.txt
python -m spacy download en_core_web_sm
python pipeline_eda.py          # (opcional) regenera los CSV preprocesados del Módulo 2
python clasificador_tfidf.py    # corre el clasificador del Módulo 3
```