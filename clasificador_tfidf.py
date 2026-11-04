"""
Clasificador Supervisado con TF-IDF - Módulo 3
Implementación de pipeline de clasificación de texto usando técnicas de NLP clásicas.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

# Scikit-learn imports
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score
)
from sklearn.pipeline import Pipeline
import joblib

# Local imports
from cargar_dataset import cargar

# Configuración
OUTPUT_DIR = Path(__file__).parent / "salidas_mod3"
OUTPUT_DIR.mkdir(exist_ok=True)

# Etiquetas reales del CSV -> nombre legible para reportes
LABEL_NAMES = {
    "World": "World",
    "Sports": "Sports",
    "Business": "Business",
    "Sci_Tech": "Sci/Tech",
}


def get_label_order(y):
    """Devuelve las etiquetas en el MISMO orden que usa sklearn (ordenadas)."""
    return sorted(pd.Series(y).unique())


def get_display_names(y):
    """Nombres legibles alineados con el orden interno de sklearn."""
    return [LABEL_NAMES.get(lbl, lbl) for lbl in get_label_order(y)]

def load_preprocessed_data():
    """
    Carga los datos preprocesados del módulo anterior.
    Si no existen, ejecuta el preprocesamiento.
    """
    preproc_train = Path("salidas") / "ag_news_train_preprocesado.csv"
    preproc_test = Path("salidas") / "ag_news_test_preprocesado.csv"
    
    if preproc_train.exists() and preproc_test.exists():
        print("Cargando datos preprocesados del módulo 2...")
        train_df = pd.read_csv(preproc_train)
        test_df = pd.read_csv(preproc_test)
    else:
        print("Ejecutando preprocesamiento del módulo 2...")
        # Importar y ejecutar pipeline de preprocesamiento
        from pipeline_eda import main as preprocess_main
        preprocess_main()
        
        train_df = pd.read_csv(preproc_train)
        test_df = pd.read_csv(preproc_test)
    
    # Verificar que tenemos la columna 'clean'
    if 'clean' not in train_df.columns or 'clean' not in test_df.columns:
        raise ValueError("Los datos preprocesados no contienen la columna 'clean'")
    
    return train_df, test_df

def create_tfidf_vectorizer(max_features=10000, ngram_range=(1, 2), **kwargs):
    """
    Crea un vectorizador TF-IDF con parámetros configurables.
    
    Args:
        max_features: Máximo número de features (vocabulario)
        ngram_range: Rango de n-gramas a considerar
        **kwargs: Parámetros adicionales para TfidfVectorizer
    
    Returns:
        TfidfVectorizer configurado
    """
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english',  # Filtra stop-words en inglés
        min_df=2,  # Ignora términos que aparecen en menos de 2 documentos
        max_df=0.95,  # Ignora términos que aparecen en más del 95% de documentos
        **kwargs
    )

def evaluate_model(model, X_test, y_test, model_name, save_results=True):
    """
    Evalúa un modelo y genera métricas de evaluación.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba vectorizados
        y_test: Etiquetas de prueba
        model_name: Nombre del modelo para reportes
        save_results: Si True, guarda resultados en archivos
    
    Returns:
        dict con métricas de evaluación
    """
    # Predicciones
    y_pred = model.predict(X_test)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average='macro')
    f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    # Reporte de clasificación (nombres alineados con el orden interno de sklearn)
    display_names = get_display_names(y_test)
    report = classification_report(
        y_test, 
        y_pred, 
        labels=get_label_order(y_test),
        target_names=display_names,
        output_dict=True
    )
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Resultados
    results = {
        'model_name': model_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'timestamp': datetime.now().isoformat()
    }
    
    if save_results:
        # Guardar reporte como JSON
        report_path = OUTPUT_DIR / f"{model_name}_report.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Guardar matriz de confusión como imagen
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm, 
            annot=True, 
            fmt='d',
            cmap='Blues',
            xticklabels=display_names,
            yticklabels=display_names
        )
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Etiqueta Real')
        plt.xlabel('Etiqueta Predicha')
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{model_name}_confusion_matrix.png", dpi=150)
        plt.close()
        
        # Guardar reporte de clasificación como texto
        text_report = classification_report(
            y_test, 
            y_pred, 
            labels=get_label_order(y_test),
            target_names=display_names
        )
        with open(OUTPUT_DIR / f"{model_name}_classification_report.txt", 'w', encoding='utf-8') as f:
            f.write(f"Modelo: {model_name}\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"F1 Macro: {f1_macro:.4f}\n")
            f.write(f"F1 Weighted: {f1_weighted:.4f}\n\n")
            f.write("Reporte de Clasificación:\n")
            f.write(text_report)
    
    return results

def train_baseline_models(X_train, y_train, X_test, y_test, vectorizer):
    """
    Entrena y evalúa modelos baseline.
    
    Args:
        X_train: Textos de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_test: Textos de prueba
        y_test: Etiquetas de prueba
        vectorizer: Vectorizador TF-IDF ya ajustado a X_train
    
    Returns:
        dict con resultados de todos los modelos
    """
    results = {}
    
    # Modelos a probar
    models = {
        'NaiveBayes': MultinomialNB(),
        'SVM': LinearSVC(random_state=42, max_iter=10000),
        'LogisticRegression': LogisticRegression(random_state=42, max_iter=10000)
    }
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Entrenando {name}...")
        
        # Entrenar modelo
        model.fit(X_train, y_train)
        
        # Evaluar
        model_results = evaluate_model(model, X_test, y_test, name)
        results[name] = model_results
        
        # Guardar modelo
        model_path = OUTPUT_DIR / f"{name}_model.pkl"
        joblib.dump(model, model_path)
        
        # Cross-validation (opcional, para análisis más profundo)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
        results[name]['cv_f1_macro_mean'] = cv_scores.mean()
        results[name]['cv_f1_macro_std'] = cv_scores.std()
        
        print(f"  Accuracy: {model_results['accuracy']:.4f}")
        print(f"  F1 Macro: {model_results['f1_macro']:.4f}")
        print(f"  CV F1 Macro: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    
    return results

def analyze_feature_importance(vectorizer, model, feature_names, top_n=20):
    """
    Analiza la importancia de features para modelos lineales.
    
    Args:
        vectorizer: Vectorizador TF-IDF
        model: Modelo entrenado
        feature_names: Nombres de las features
        top_n: Número de features más importantes a mostrar
    """
    if hasattr(model, 'coef_'):
        # Para modelos lineales (SVM, Logistic Regression)
        coef = model.coef_
        
        # Las filas de coef_ siguen el orden de model.classes_
        class_labels = list(model.classes_)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, label in enumerate(class_labels[:4]):
            category_name = LABEL_NAMES.get(str(label), str(label))
            
            # Obtener coeficientes para esta categoría
            if len(coef.shape) == 1:
                # Para clasificación binaria
                coef_i = coef
            else:
                # Para clasificación multiclase
                coef_i = coef[i]
            
            # Obtener top_n features positivas y negativas
            top_pos_idx = np.argsort(coef_i)[-top_n:]
            top_neg_idx = np.argsort(coef_i)[:top_n]
            
            top_features = list(top_neg_idx) + list(top_pos_idx)
            top_coefs = coef_i[top_features]
            top_names = [feature_names[j] for j in top_features]
            
            # Crear gráfico
            colors = ['red' if c < 0 else 'blue' for c in top_coefs]
            axes[i].barh(range(len(top_coefs)), top_coefs, color=colors)
            axes[i].set_yticks(range(len(top_coefs)))
            axes[i].set_yticklabels(top_names)
            axes[i].set_title(f'Top {top_n*2} Features - {category_name}')
            axes[i].set_xlabel('Coeficiente')
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=150, bbox_inches='tight')
        plt.close()

def experiment_with_parameters(X_train, y_train, X_test, y_test):
    """
    Experimenta con diferentes parámetros del vectorizador TF-IDF.
    
    Args:
        X_train: Textos de entrenamiento
        y_train: Etiquetas de entrenamiento
        X_test: Textos de prueba
        y_test: Etiquetas de prueba
    
    Returns:
        dict con resultados de experimentos
    """
    experiments = {}
    
    # Experimentos con diferentes configuraciones
    configs = [
        {'max_features': 5000, 'ngram_range': (1, 1), 'name': 'unigram_5k'},
        {'max_features': 10000, 'ngram_range': (1, 1), 'name': 'unigram_10k'},
        {'max_features': 5000, 'ngram_range': (1, 2), 'name': 'bigram_5k'},
        {'max_features': 10000, 'ngram_range': (1, 2), 'name': 'bigram_10k'},
        {'max_features': 20000, 'ngram_range': (1, 2), 'name': 'bigram_20k'},
    ]
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Experimentando con: {config['name']}")
        
        # El vectorizador va DENTRO del Pipeline para que la validación cruzada
        # lo ajuste en cada fold de train y no vea nunca los datos de test.
        pipe = Pipeline([
            ('tfidf', create_tfidf_vectorizer(
                max_features=config['max_features'],
                ngram_range=config['ngram_range']
            )),
            ('clf', LogisticRegression(random_state=42, max_iter=10000)),
        ])
        
        # Selección de configuración SOLO con validación cruzada sobre train
        cv_scores = cross_val_score(pipe, X_train, y_train, cv=3, scoring='f1_macro')
        
        # Métricas de test solo informativas (no se usan para elegir la config)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1_macro = f1_score(y_test, y_pred, average='macro')
        vocab_size = len(pipe.named_steps['tfidf'].vocabulary_)
        
        experiments[config['name']] = {
            'config': config,
            'cv_f1_macro_mean': float(cv_scores.mean()),
            'cv_f1_macro_std': float(cv_scores.std()),
            'test_accuracy': accuracy,
            'test_f1_macro': f1_macro,
            'vocabulary_size': vocab_size,
            'feature_names_sample': list(pipe.named_steps['tfidf'].get_feature_names_out()[:10])
        }
        
        print(f"  Vocabulary size: {vocab_size}")
        print(f"  CV F1 Macro (train): {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        print(f"  Test Accuracy (informativo): {accuracy:.4f}")
        print(f"  Test F1 Macro (informativo): {f1_macro:.4f}")
    
    # Guardar resultados de experimentos
    with open(OUTPUT_DIR / "tfidf_experiments.json", 'w', encoding='utf-8') as f:
        json.dump(experiments, f, indent=2, default=str)
    
    return experiments

def main():
    """Pipeline principal de clasificación supervisada."""
    print("=" * 60)
    print("CLASIFICADOR SUPERVISADO CON TF-IDF - MÓDULO 3")
    print("=" * 60)
    
    # 1. Cargar datos preprocesados
    print("\n1. Cargando datos preprocesados...")
    train_df, test_df = load_preprocessed_data()
    
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    print(f"   Categorías: {list(LABEL_NAMES.values())}")
    
    # 2. Preparar datos
    X_train = train_df['clean'].fillna('')
    y_train = train_df['label']
    X_test = test_df['clean'].fillna('')
    y_test = test_df['label']
    
    # 3. Experimentar con diferentes parámetros de TF-IDF
    print("\n2. Experimentando con parámetros de TF-IDF...")
    experiments = experiment_with_parameters(X_train, y_train, X_test, y_test)
    
    # 4. Configuración final (mejor configuración basada en experimentos)
    print("\n3. Entrenando modelos baseline con mejor configuración...")
    
    # Seleccionar configuración por CV en train (nunca por performance en test)
    best_config_name = max(experiments.items(), key=lambda x: x[1]['cv_f1_macro_mean'])[0]
    print(f"   Mejor configuración (por CV F1 Macro en train): {best_config_name}")
    print(f"   CV F1 Macro: {experiments[best_config_name]['cv_f1_macro_mean']:.4f}")
    
    # Crear vectorizador con mejor configuración
    vectorizer = create_tfidf_vectorizer(
        max_features=experiments[best_config_name]['config']['max_features'],
        ngram_range=experiments[best_config_name]['config']['ngram_range']
    )
    
    # Ajustar SOLO en datos de entrenamiento (evitar data leakage)
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"   Train matrix shape: {X_train_vec.shape}")
    print(f"   Test matrix shape: {X_test_vec.shape}")
    
    # Guardar vectorizador
    vectorizer_path = OUTPUT_DIR / "tfidf_vectorizer.pkl"
    joblib.dump(vectorizer, vectorizer_path)
    
    # 5. Entrenar y evaluar modelos baseline
    results = train_baseline_models(X_train_vec, y_train, X_test_vec, y_test, vectorizer)
    
    # 6. Análisis de importancia de features (para modelos lineales)
    print("\n4. Analizando importancia de features...")
    feature_names = vectorizer.get_feature_names_out()
    
    # Cargar mejor modelo (basado en F1 Macro)
    best_model_name = max(results.items(), key=lambda x: x[1]['f1_macro'])[0]
    best_model = joblib.load(OUTPUT_DIR / f"{best_model_name}_model.pkl")
    
    # La importancia de features solo existe en modelos lineales,
    # así que se usa el mejor modelo que exponga coef_ (SVM o Regresión Logística).
    importance_model_name = next(
        (name for name in sorted(results, key=lambda n: results[n]['f1_macro'], reverse=True)
         if hasattr(joblib.load(OUTPUT_DIR / f"{name}_model.pkl"), 'coef_')),
        None
    )
    if importance_model_name:
        importance_model = joblib.load(OUTPUT_DIR / f"{importance_model_name}_model.pkl")
        analyze_feature_importance(vectorizer, importance_model, feature_names)
        print(f"   Importancia de features calculada con {importance_model_name}")
        print(f"   Gráfico guardado en: {OUTPUT_DIR / 'feature_importance.png'}")
    
    # Guardar el pipeline completo (vectorizador + mejor modelo) listo para producción
    final_pipeline = Pipeline([('tfidf', vectorizer), ('clf', best_model)])
    pipeline_path = OUTPUT_DIR / "pipeline_completo.pkl"
    joblib.dump(final_pipeline, pipeline_path)
    print(f"   Pipeline completo guardado en: {pipeline_path}")
    
    # 7. Resumen de resultados
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    
    summary = []
    for model_name, model_results in results.items():
        summary.append({
            'model': model_name,
            'accuracy': model_results['accuracy'],
            'f1_macro': model_results['f1_macro'],
            'f1_weighted': model_results['f1_weighted']
        })
    
    summary_df = pd.DataFrame(summary)
    print("\nComparación de modelos:")
    print(summary_df.to_string(index=False))
    
    # Guardar resumen
    summary_path = OUTPUT_DIR / "model_comparison.csv"
    summary_df.to_csv(summary_path, index=False)
    
    # 8. Identificar categorías más difíciles
    print("\n5. Análisis de categorías más difíciles...")
    
    # Usar mejor modelo para análisis detallado
    y_pred = best_model.predict(X_test_vec)
    report = classification_report(
        y_test, y_pred,
        labels=get_label_order(y_test),
        target_names=get_display_names(y_test),
        output_dict=True
    )
    
    # Extraer métricas por clase
    class_metrics = []
    for label in get_label_order(y_test):
        category_name = LABEL_NAMES.get(str(label), str(label))
        if category_name in report:
            metrics = report[category_name]
            class_metrics.append({
                'category': category_name,
                'precision': metrics['precision'],
                'recall': metrics['recall'],
                'f1_score': metrics['f1-score'],
                'support': metrics['support']
            })
    
    class_df = pd.DataFrame(class_metrics)
    print("\nMétricas por categoría:")
    print(class_df.to_string(index=False))
    
    # Identificar categoría más difícil (menor F1-score)
    hardest_category = class_df.loc[class_df['f1_score'].idxmin()]
    print(f"\nCategoría más difícil: {hardest_category['category']} (F1-score: {hardest_category['f1_score']:.4f})")
    
    # Guardar análisis por categoría
    class_path = OUTPUT_DIR / "category_analysis.csv"
    class_df.to_csv(class_path, index=False)
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETADO EXITOSAMENTE")
    print(f"Resultados guardados en: {OUTPUT_DIR}")
    print("=" * 60)
    
    return {
        'experiments': experiments,
        'model_results': results,
        'best_model': best_model_name,
        'hardest_category': hardest_category['category'],
        'output_dir': str(OUTPUT_DIR)
    }

if __name__ == "__main__":
    main()