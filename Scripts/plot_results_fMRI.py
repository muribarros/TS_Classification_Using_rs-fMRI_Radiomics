#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:51:04 2026

@author: Lab
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# ==================== DADOS ====================
# Acurácias (conforme sua tabela)
modelos = ['Naive Bayes', 'SVM Linear', 'SVM RBF', 'MLP-3L', 'MLP-4L', 'MLP-5L']
acc = [71.21, 72.73, 72.73, 78.00, 71.21, 72.73]
# Desvios padrão (estimados, pois sua tabela não mostra; você pode usar erro padrão da CV)
# Se não tiver, coloque 0 ou um valor pequeno como 1.5
std_acc = [1.5, 1.2, 1.2, 1.8, 2.0, 1.5]

# Frequência de seleção das regiões (contei manualmente da sua tabela)
# Lista de regiões e quantos modelos (dentre os 6) a selecionaram
regioes = [
    'Left Accumbens', 'Left Caudate', 'Left Thalamus',
    'Right Caudate', 'Left Lateral OFC', 'Right Thalamus',
    'Right Ventral DC', 'Left Medial OFC', 'Left Putamen',
    'Left Ventral DC', 'Right Medial OFC'
]
freq = [5, 4, 4, 3, 3, 2, 2, 1, 1, 1, 1]

# Ordenar por frequência decrescente
sorted_idx = np.argsort(freq)[::-1]
regioes_sorted = [regioes[i] for i in sorted_idx]
freq_sorted = [freq[i] for i in sorted_idx]

# ==================== CRIAÇÃO DA FIGURA ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# --- Painel A: Acurácias ---
cores = ['#4c72b0']*6
cores[3] = '#dd8452'  # destaca MLP-3L
bars = ax1.bar(modelos, acc, yerr=std_acc, capsize=5, color=cores, edgecolor='black')
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(65, 85)
ax1.set_title('(A) Classifier Performance (5-fold CV)')
ax1.grid(axis='y', linestyle='--', alpha=0.5)
# Adicionar valores no topo das barras
for bar, a in zip(bars, acc):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{a:.1f}%',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax1.tick_params(axis='x', rotation=20)

# --- Painel B: Frequência de seleção (barras horizontais) ---
sns.barplot(y=regioes_sorted, x=freq_sorted, palette='viridis_r', ax=ax2, hue=regioes_sorted, legend=False)
ax2.set_xlabel('Selection Frequency (out of 6 models)')
ax2.set_title('(B) Regional Consistency Across Classifiers')
ax2.set_xlim(0, 6)
ax2.grid(axis='x', linestyle='--', alpha=0.5)
# Adicionar números
for i, (reg, f) in enumerate(zip(regioes_sorted, freq_sorted)):
    ax2.text(f + 0.1, i, f'{f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('figure_results.eps', dpi=300, bbox_inches='tight')
plt.show()