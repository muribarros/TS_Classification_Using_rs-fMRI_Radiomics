#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 11:36:35 2026

@author: Lab
"""

import pandas as pd
import numpy as np

# --- Monkey Patch - Correção para possível ERRO DO NUMPY ---
np.object = object
np.bool = bool
np.int = int
np.float = float
# ------------------------------------------------------

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn_genetic import GAFeatureSelectionCV
from sklearn_genetic.callbacks.base import BaseCallback
from sklearn.base import BaseEstimator, TransformerMixin
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore")

print("=========================================================")
print("  BUSCA GENÉTICA (Evolução Contínua com Bloco de Métricas)")
print("=========================================================")

# ------------------------------------------------------
# 1. CARREGAMENTO DOS DADOS
# ------------------------------------------------------
dir_glcm = "/Users/Lab/Documents/Murilo/DataSet_RM/GLCM_in_fMRI_TESTE/"
df = pd.read_csv(dir_glcm + 'Dataset_GLCM_Sem_Outliers.csv')

y = df['Classe'].values
todas_features = [c for c in df.columns if c not in ['Paciente', 'Classe']]
todas_regioes = sorted(list(set(["_".join(f.split("_")[:-1]) for f in todas_features])))

# ------------------------------------------------------
# 2. PREPARANDO OS DADOS EM 3D E A MATRIZ "DUMMY" (O TRUQUE)
# ------------------------------------------------------
n_pacientes = len(df)
n_regioes = len(todas_regioes)

# A. Agrupando as features reais num bloco 3D (Paciente x Região x 24 Métricas)
X_full_list = []
for i in range(n_pacientes):
    paciente_data = []
    for reg in todas_regioes:
        cols_reg = [c for c in todas_features if c.startswith(reg + '_')]
        # Garante a coleta das 24 métricas exatas da região
        paciente_data.append(df.loc[i, cols_reg].values.astype(float))
    X_full_list.append(paciente_data)
    
X_full_3d = np.array(X_full_list) # Matriz Real: (66, 17, 24)

# B. Criando o X_dummy (O "Código de Barras" para o GA selecionar regiões)
X_dummy = np.zeros((n_pacientes, n_regioes), dtype=int)
for i in range(n_pacientes):
    for j in range(n_regioes):
        # Gera um ID único: Ex: Paciente 5, Região 12 vira 500012
        X_dummy[i, j] = (i * 100000) + j 

# ------------------------------------------------------
# 3. TRANSFORMADOR E MONITOR CUSTOMIZADOS
# ------------------------------------------------------
class ExpansorDeRegioes(BaseEstimator, TransformerMixin):
    """
    Este transformador pega as regiões selecionadas pelo GA (via X_dummy)
    e as expande, trazendo todas as 24 métricas de GLCM para a Rede Neural.
    """
    def __init__(self, X_real_3d):
        self.X_real_3d = X_real_3d
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        n_samples = X.shape[0]
        n_selected_regions = X.shape[1]
        
        # Proteção caso o GA tente uma mutação que desliga TODAS as regiões
        if n_selected_regions == 0:
            return np.zeros((n_samples, 1))
            
        # Decodifica os "Códigos de Barra"
        paciente_indices = (X[:, 0] // 100000).astype(int)
        regiao_indices = (X[0, :] % 100000).astype(int)
        
        # Busca no banco 3D os pacientes e as regiões ativas (trazendo as 24 métricas juntas)
        X_expandido = self.X_real_3d[np.ix_(paciente_indices, regiao_indices)]
        
        # Achata de (Amostras, Regiões, 24) para (Amostras, Regiões * 24)
        return X_expandido.reshape(n_samples, -1)

class MonitorRegioes(BaseCallback):
    """
    Inspeciona a evolução e imprime a quantidade de regiões usadas pelo melhor modelo.
    """
    def __init__(self):
        self.geracao = 1
        
    def on_step(self, record, logbook, estimator):
        try:
            if hasattr(estimator, 'best_features_'):
                qtd_regioes = int(sum(estimator.best_features_))
                print(f"   [Info Genética] Vencedor provisório da Gen {self.geracao:02d} utiliza {qtd_regioes} regiões ({qtd_regioes * 24} features).")
        except Exception:
            pass
        self.geracao += 1

# ------------------------------------------------------
# 4. TIPOS DE ARQUITETURAS
# ------------------------------------------------------
modelos = {
    'Naive_Bayes': GaussianNB(),
    'SVM_Linear': SVC(kernel='linear', random_state=42),
    'SVM_RBF': SVC(kernel='rbf', C=3.0, gamma='scale', random_state=42),
    'MLP_3_Layers': MLPClassifier(hidden_layer_sizes=(16, 8, 4), max_iter=1000, random_state=42),
    'MLP_4_Layers': MLPClassifier(hidden_layer_sizes=(16, 12, 8, 4), max_iter=1000, random_state=42),
    'MLP_5_Layers': MLPClassifier(hidden_layer_sizes=(16, 12, 8, 4, 2), max_iter=1000, random_state=42)
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
resultados_tabela = []

# ------------------------------------------------------
# 5. EXECUTANDO A EVOLUÇÃO PARA CADA MODELO
# ------------------------------------------------------
for nome_modelo, classificador in modelos.items():
    print(f"\n-> [Treinando {nome_modelo}] Acompanhe a coluna 'fitness_max'...")
    
    # ATENÇÃO: Adicionamos o nosso Expansor como a 1ª etapa do Pipeline!
    pipe = Pipeline([
        ("expansor", ExpansorDeRegioes(X_full_3d)),
        ("scaler", QuantileTransformer(n_quantiles=50, output_distribution='normal', random_state=42)),
        ("clf", classificador)
    ])
    
    evolucao = GAFeatureSelectionCV(
        estimator=pipe,
        cv=cv,
        scoring="accuracy",
        population_size=50,       
        generations=100,           
        n_jobs=3,                 
        verbose=True,             
        keep_top_k=5,             
        mutation_probability=0.2, 
    )

    # Inicia a evolução na Matriz "Falsa" (X_dummy), e aplica o Monitor
    evolucao.fit(X_dummy, y, callbacks=[MonitorRegioes()])
    
    # Extrai o DNA Vencedor Final
    features_selecionadas_idx = evolucao.best_features_
    regioes_vencedoras = [todas_regioes[i] for i, mask in enumerate(features_selecionadas_idx) if mask]
    
    # RECALCULANDO A ACURÁCIA
    # Passa o X_dummy mascarado. O Pipeline vai convertê-lo para as métricas reais via Expansor.
    X_vencedor = X_dummy[:, features_selecionadas_idx]
    predicoes = np.zeros(len(y))
    
    for train_idx, test_idx in cv.split(X_vencedor, y):
        X_tr, X_te = X_vencedor[train_idx], X_vencedor[test_idx]
        y_tr = y[train_idx]
        pipe.fit(X_tr, y_tr)
        predicoes[test_idx] = pipe.predict(X_te)
        
    acc_final = accuracy_score(y, predicoes)
    
    resultados_tabela.append({
        'Modelo': nome_modelo,
        'Qtd_Regioes': len(regioes_vencedoras),
        'Qtd_Total_Features': len(regioes_vencedoras) * 24, # Evidencia o impacto das 24 métricas
        'Acuracia': acc_final,
        'Regioes': " + ".join(regioes_vencedoras)
    })
    
    print(f"\n   >>> VENCEDOR [{nome_modelo}]: {acc_final * 100:.2f}% com {len(regioes_vencedoras)} regiões ({len(regioes_vencedoras)*24} features).")

# ------------------------------------------------------
# 6. EXIBINDO OS RESULTADOS FINAIS
# ------------------------------------------------------
df_tabela = pd.DataFrame(resultados_tabela)
df_tabela['Acuracia'] = (df_tabela['Acuracia'] * 100).round(2)

pd.set_option('display.max_colwidth', None)
pd.set_option('display.width', 1000)

print("\n=========================================================")
print(" RESULTADO FINAL: BUSCA GENÉTICA (Todas as Métricas GLCM)")
print("=========================================================")
print(df_tabela[['Modelo', 'Qtd_Regioes', 'Qtd_Total_Features', 'Acuracia', 'Regioes']])

caminho_csv = dir_glcm + 'Resultados_Geneticos_Kaue2.csv'
df_tabela.to_csv(caminho_csv, index=False)
print(f"\n-> Arquivo detalhado salvo em: {caminho_csv}")