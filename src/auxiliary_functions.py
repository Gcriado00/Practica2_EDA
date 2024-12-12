import pandas as pd 
import numpy as np
import sklearn
from sklearn.pipeline import Pipeline
from sklearn import metrics
import category_encoders as ce
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.preprocessing import StandardScaler

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, \
                            silhouette_score, recall_score, precision_score, make_scorer, \
                            roc_auc_score, f1_score, precision_recall_curve, accuracy_score, roc_auc_score, \
                            classification_report, confusion_matrix

from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay  
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from catboost import CatBoostClassifier
from collections import Counter
from sklearn.inspection import PartialDependenceDisplay

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_confusion_matrices(cm, cm_normalized, classifier_name, y_test_labels):
    """
    Graficar las matrices de confusión (normal y normalizada).
    
    Args:
        cm (array-like): Matriz de confusión normal.
        cm_normalized (array-like): Matriz de confusión normalizada.
        classifier_name (str): Nombre del clasificador.
        y_test_labels (array-like): Etiquetas únicas de la clase real.
    """
    fig, axes = plt.subplots(1, 2, figsize=(7, 4))

    # Matriz de confusión normal
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test_labels)
    disp.plot(ax=axes[0], cmap=plt.cm.Blues, colorbar=False)
    axes[0].set_title(f"Normal ({classifier_name})")
    axes[0].set_xlabel("Predicción")
    axes[0].set_ylabel("Clase Real")

    # Matriz de confusión normalizada
    disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=y_test_labels)
    disp_norm.plot(ax=axes[1], cmap=plt.cm.Oranges, colorbar=False)
    axes[1].set_title(f"Normalizada ({classifier_name})")
    axes[1].set_xlabel("Predicción")
    axes[1].set_ylabel("Clase Real")

    # Ajustar diseño y mostrar
    plt.tight_layout()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_feature_importances(feature_importances, feature_names, title="Importancia de características en AdaBoost"):
    """
    Grafica las importancias de características en un modelo basado en árboles.

    Args:
        feature_importances (array-like): Importancias de las características.
        feature_names (array-like): Nombres de las características.
        title (str): Título del gráfico. Por defecto: "Importancia de características en AdaBoost".
    """
    # Ordenar por importancia
    sorted_indices = np.argsort(feature_importances)[::-1]
    sorted_importances = feature_importances[sorted_indices]
    sorted_features = feature_names[sorted_indices]

    # Crear el gráfico
    plt.figure(figsize=(10, 8))
    plt.barh(sorted_features, sorted_importances, color="skyblue")
    plt.xlabel("Importancia de la característica")
    plt.ylabel("Características")
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.show()

# Ejemplo de uso
feature_importances = np.array([0.005, 0.0, 0.0, 0.005, 0.015, 0.0, 0.035, 0.0, 0.0,
                                 0.015, 0.0, 0.015, 0.01, 0.0, 0.095, 0.015, 0.115, 0.025,
                                 0.05, 0.05, 0.0, 0.0, 0.035, 0.0, 0.0, 0.025, 0.01,
                                 0.135, 0.005, 0.0, 0.0, 0.155, 0.0, 0.045, 0.04, 0.015,
                                 0.0, 0.0, 0.0, 0.0, 0.0, 0.085])

feature_names = np.array(['CODE_GENDER_1', 'CODE_GENDER_2', 'FLAG_OWN_CAR_1',
                          'FLAG_OWN_CAR_2', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
                          'NAME_INCOME_TYPE_6', 'NAME_INCOME_TYPE_7',
                          'NAME_EDUCATION_TYPE_1', 'NAME_EDUCATION_TYPE_2', 'DAYS_EMPLOYED',
                          'DAYS_ID_PUBLISH', 'FLAG_WORK_PHONE', 'OCCUPATION_TYPE',
                          'REGION_RATING_CLIENT_W_CITY', 'ORGANIZATION_TYPE', 'EXT_SOURCE_1',
                          'EXT_SOURCE_2', 'EXT_SOURCE_3', 'BASEMENTAREA_AVG',
                          'YEARS_BUILD_AVG', 'NONLIVINGAREA_AVG', 'YEARS_BUILD_MODE',
                          'ELEVATORS_MODE', 'FLOORSMAX_MODE', 'NONLIVINGAPARTMENTS_MODE',
                          'NONLIVINGAREA_MODE', 'COMMONAREA_MEDI', 'LIVINGAPARTMENTS_MEDI',
                          'NONLIVINGAPARTMENTS_MEDI', 'OBS_30_CNT_SOCIAL_CIRCLE',
                          'DEF_30_CNT_SOCIAL_CIRCLE', 'OBS_60_CNT_SOCIAL_CIRCLE',
                          'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_3', 'FLAG_DOCUMENT_4',
                          'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_10',
                          'FLAG_DOCUMENT_18', 'AMT_REQ_CREDIT_BUREAU_QRT'])

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_precision_recall_curve(y_test, y_pred_proba, model_name="Modelo"):
    """
    Calcula y grafica la curva Precision-Recall para un modelo clasificador.

    Args:
        y_test (array-like): Etiquetas reales del conjunto de prueba.
        y_pred_proba (array-like): Probabilidades predichas para la clase positiva.
        model_name (str): Nombre del modelo para incluir en el título del gráfico.
    """
    # Calcular la curva Precision-Recall
    precision, recall, thresholds = precision_recall_curve(y_test, y_pred_proba)

    # Calcular el área bajo la curva (AUC-PR)
    pr_auc = auc(recall, precision)
    print(f"AUC-PR: {pr_auc:.3f}")

    # Encontrar el mejor punto en la curva Precision-Recall (mayor F1-score)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-6)
    best_index = np.argmax(f1_scores)
    print(f"Mejor umbral: {thresholds[best_index]:.3f}, Precision: {precision[best_index]:.3f}, Recall: {recall[best_index]:.3f}")

    # Graficar la curva Precision-Recall
    plt.figure(figsize=(7, 4))
    plt.plot(recall, precision, marker='.', label=f'{model_name} (AUC = {pr_auc:.3f})')  # Curva Precision-Recall
    plt.scatter(recall[best_index], precision[best_index], s=100, marker='o', color='black', label='Best')  # Mejor punto

    # Etiquetas de los ejes
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_roc_curve(y_test, y_pred_proba, model_name="Modelo"):
    """
    Calcula y grafica la curva ROC para un modelo clasificador.

    Args:
        y_test (array-like): Etiquetas reales del conjunto de prueba.
        y_pred_proba (array-like): Probabilidades predichas para la clase positiva.
        model_name (str): Nombre del modelo para incluir en el título del gráfico.
    """
    # Calcular la curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    
    # Calcular el AUC-ROC
    auc_value = roc_auc_score(y_test, y_pred_proba)
    
    # Encontrar el mejor punto en la curva ROC (punto más cercano a la esquina superior izquierda)
    gmeans = np.sqrt(tpr * (1 - fpr))
    ix = np.argmax(gmeans)
    print(f"Mejor umbral: {thresholds[ix]:.3f}, TPR: {tpr[ix]:.3f}, FPR: {fpr[ix]:.3f}")

    # Graficar la curva ROC
    plt.figure(figsize=(5, 4))
    plt.plot([0, 1], [0, 1], linestyle='--', label='No Skill')
    plt.plot(fpr, tpr, marker='.', label=f'{model_name} (AUC = {auc_value:.3f})')
    plt.scatter(fpr[ix], tpr[ix], s=100, marker='o', color='black', label='Best') 

    # Etiquetas de los ejes
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_lift_curve_by_class(y_test, y_pred_proba, model_name="Modelo"):
    """
    Calcula y grafica la curva LIFT separada por clases para un modelo clasificador.

    Args:
        y_test (array-like): Etiquetas reales del conjunto de prueba.
        y_pred_proba (array-like): Probabilidades predichas para la clase positiva.
        model_name (str): Nombre del modelo para incluir en el título del gráfico.
    """
  

    # Crear un DataFrame con etiquetas reales y probabilidades predichas
    data = pd.DataFrame({'y_test': y_test, 'y_pred_proba': y_pred_proba})
    data = data.sort_values(by='y_pred_proba', ascending=False)

    # Calcular lift por clase
    results = []
    total_positives = (data['y_test'] == 1).sum()
    total_negatives = (data['y_test'] == 0).sum()
    cumulative_positives = data['y_test'].cumsum()
    cumulative_negatives = (~data['y_test'].astype(bool)).cumsum()

    lift_positive = cumulative_positives / total_positives
    lift_negative = cumulative_negatives / total_negatives

    # Construir DataFrame para graficar
    lift_df = pd.DataFrame({
        'Porcentaje de la muestra': np.arange(1, len(data) + 1) / len(data),
        'Clase 1 (Lift)': lift_positive,
        'Clase 0 (Lift)': lift_negative,
    })

    # Graficar
    plt.figure(figsize=(7, 4))
    sns.lineplot(x='Porcentaje de la muestra', y='Clase 1 (Lift)', data=lift_df, label='Clase 1', color='orange')
    sns.lineplot(x='Porcentaje de la muestra', y='Clase 0 (Lift)', data=lift_df, label='Clase 0', color='blue')
    plt.axhline(y=1, color='black', linestyle='--', label='Línea base')
    plt.xlabel('Porcentaje de la muestra')
    plt.ylabel('Lift')
    plt.title(f'Curva Lift - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------#

def plot_cumulative_gains_by_class(y_test, y_pred_proba, model_name="Modelo"):
    """
    Calcula y grafica la curva de ganancias acumuladas (Cumulative Gains) separada por clases para un modelo clasificador.

    Args:
        y_test (array-like): Etiquetas reales del conjunto de prueba.
        y_pred_proba (array-like): Probabilidades predichas para la clase positiva.
        model_name (str): Nombre del modelo para incluir en el título del gráfico.
    """

    # Crear un DataFrame con etiquetas reales y probabilidades predichas
    data = pd.DataFrame({'y_test': y_test, 'y_pred_proba': y_pred_proba})
    data = data.sort_values(by='y_pred_proba', ascending=False)

    # Calcular ganancias acumuladas por clase
    cumulative_positives = data['y_test'].cumsum()
    cumulative_negatives = (~data['y_test'].astype(bool)).cumsum()
    total_positives = (data['y_test'] == 1).sum()
    total_negatives = (data['y_test'] == 0).sum()

    gains_positive = cumulative_positives / total_positives
    gains_negative = cumulative_negatives / total_negatives

    # Construir DataFrame para graficar
    gains_df = pd.DataFrame({
        'Porcentaje de la muestra': np.arange(1, len(data) + 1) / len(data),
        'Clase 1 (Gain)': gains_positive,
        'Clase 0 (Gain)': gains_negative,
    })

    # Graficar
    plt.figure(figsize=(7, 4))
    sns.lineplot(x='Porcentaje de la muestra', y='Clase 1 (Gain)', data=gains_df, label='Clase 1', color='orange')
    sns.lineplot(x='Porcentaje de la muestra', y='Clase 0 (Gain)', data=gains_df, label='Clase 0', color='blue')
    plt.plot([0, 1], [0, 1], linestyle='--', color='black', label='Línea base')
    plt.xlabel('Porcentaje de la muestra')
    plt.ylabel('Ganancia acumulada')
    plt.title(f'Curva de Ganancias Acumuladas - {model_name}')
    plt.legend()
    plt.grid()
    plt.show()


