#!/usr/bin/env python3
"""
ANALISI DEL BIAS NEL DATASET ADULT INCOME
Dataset pubblico sui redditi negli Stati Uniti per studiare la predizione di guadagni >50K

Gruppi sensibili: Genere (sex), etnia (race), età
Analisi: conta esempi per gruppo, calcola % ">50K", identifica squilibri
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_and_prepare_data(file_path):
    """Carica e prepara il dataset Adult Income"""
    
    # Nomi delle colonne standard del dataset Adult Income
    column_names = [
        'age', 'workclass', 'fnlwgt', 'education', 'education_num',
        'marital_status', 'occupation', 'relationship', 'race', 'sex',
        'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
    ]
    
    # Carica il dataset
    df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)
    
    # Rimuovi spazi bianchi dalle stringhe
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    return df

def analyze_group_representation(df):
    """Analizza la rappresentazione dei diversi gruppi"""
    
    print("="*60)
    print("ANALISI DELLA RAPPRESENTAZIONE DEI GRUPPI")
    print("="*60)
    
    # 1. GENERE
    print("\n1. DISTRIBUZIONE PER GENERE:")
    gender_counts = df['sex'].value_counts()
    gender_perc = df['sex'].value_counts(normalize=True) * 100
    
    for gender in gender_counts.index:
        count = gender_counts[gender]
        perc = gender_perc[gender]
        print(f"   {gender}: {count:,} esempi ({perc:.1f}%)")
    
    # 2. ETNIA
    print("\n2. DISTRIBUZIONE PER ETNIA:")
    race_counts = df['race'].value_counts()
    race_perc = df['race'].value_counts(normalize=True) * 100
    
    for race in race_counts.index:
        count = race_counts[race]
        perc = race_perc[race]
        print(f"   {race}: {count:,} esempi ({perc:.1f}%)")
    
    # 3. ETÀ (per fasce)
    print("\n3. DISTRIBUZIONE PER FASCE D'ETÀ:")
    df['age_group'] = pd.cut(df['age'], 
                            bins=[0, 25, 35, 45, 55, 100], 
                            labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    
    age_counts = df['age_group'].value_counts().sort_index()
    age_perc = df['age_group'].value_counts(normalize=True).sort_index() * 100
    
    for age_group in age_counts.index:
        count = age_counts[age_group]
        perc = age_perc[age_group]
        print(f"   {age_group}: {count:,} esempi ({perc:.1f}%)")

def analyze_income_bias(df):
    """Analizza il bias nei redditi per gruppo"""
    
    print("\n" + "="*60)
    print("ANALISI DEL BIAS NEI REDDITI (>50K)")
    print("="*60)
    
    # 1. BIAS DI GENERE
    print("\n1. BIAS DI GENERE:")
    gender_income = df.groupby('sex')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    gender_counts = df.groupby('sex')['income'].apply(lambda x: (x == '>50K').sum())
    total_counts = df['sex'].value_counts()
    
    for gender in gender_income.index:
        perc_high_income = gender_income[gender]
        high_income_count = gender_counts[gender]
        total_count = total_counts[gender]
        print(f"   {gender}: {perc_high_income:.1f}% ha reddito >50K ({high_income_count:,}/{total_count:,})")
    
    # Calcola il bias ratio
    male_rate = gender_income['Male'] / 100
    female_rate = gender_income['Female'] / 100
    bias_ratio = male_rate / female_rate
    print(f"   → BIAS RATIO (M/F): {bias_ratio:.2f} (gli uomini hanno {bias_ratio:.1f}x più probabilità)")
    
    # 2. BIAS ETNICO
    print("\n2. BIAS ETNICO:")
    race_income = df.groupby('race')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    race_counts = df.groupby('race')['income'].apply(lambda x: (x == '>50K').sum())
    race_totals = df['race'].value_counts()
    
    for race in race_income.index:
        perc_high_income = race_income[race]
        high_income_count = race_counts[race]
        total_count = race_totals[race]
        print(f"   {race}: {perc_high_income:.1f}% ha reddito >50K ({high_income_count:,}/{total_count:,})")
    
    # 3. BIAS PER ETÀ
    print("\n3. BIAS PER FASCE D'ETÀ:")
    age_income = df.groupby('age_group')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    age_counts = df.groupby('age_group')['income'].apply(lambda x: (x == '>50K').sum())
    age_totals = df['age_group'].value_counts().sort_index()
    
    for age_group in age_income.index:
        perc_high_income = age_income[age_group]
        high_income_count = age_counts[age_group]
        total_count = age_totals[age_group]
        print(f"   {age_group}: {perc_high_income:.1f}% ha reddito >50K ({high_income_count:,}/{total_count:,})")

def identify_bias_issues(df):
    """Identifica specifici problemi di bias"""
    
    print("\n" + "="*60)
    print("IDENTIFICAZIONE PROBLEMI DI BIAS")
    print("="*60)
    
    # Calcola statistiche chiave
    gender_income = df.groupby('sex')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    race_income = df.groupby('race')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    
    print("\n1. SQUILIBRI IDENTIFICATI:")
    
    # Bias di genere
    male_rate = gender_income['Male']
    female_rate = gender_income['Female']
    gender_gap = male_rate - female_rate
    print(f"   • Gap di genere: {gender_gap:.1f} punti percentuali")
    print(f"     (Uomini {male_rate:.1f}% vs Donne {female_rate:.1f}%)")
    
    # Bias etnico
    white_rate = race_income['White']
    min_rate = race_income.min()
    max_rate = race_income.max()
    ethnic_gap = max_rate - min_rate
    print(f"   • Gap etnico: {ethnic_gap:.1f} punti percentuali")
    print(f"     (Range: {min_rate:.1f}% - {max_rate:.1f}%)")
    
    # Rappresentazione gruppi minoritari
    total_samples = len(df)
    white_perc = (df['race'] == 'White').mean() * 100
    male_perc = (df['sex'] == 'Male').mean() * 100
    
    print(f"   • Sovrarappresentazione White: {white_perc:.1f}% del dataset")
    print(f"   • Sovrarappresentazione Male: {male_perc:.1f}% del dataset")
    
    print("\n2. RISCHI PER UN MODELLO ML:")
    if gender_gap > 10:
        print("   • ALTO RISCHIO: Gap di genere significativo (>10%)")
    if ethnic_gap > 15:
        print("   • ALTO RISCHIO: Gap etnico significativo (>15%)")
    if white_perc > 80:
        print("   • MEDIO RISCHIO: Dataset dominato da popolazione White")
    if male_perc > 65:
        print("   • MEDIO RISCHIO: Dataset sbilanciato verso popolazione maschile")
    
    print("\n3. RACCOMANDAZIONI:")
    print("   • Utilizzare metriche fairness-aware durante il training")
    print("   • Implementare tecniche di bias mitigation (re-sampling, re-weighting)")
    print("   • Valutare le performance separatamente per ogni gruppo sensibile")
    print("   • Considerare post-processing per equalizzare gli outcome")

def create_visualizations(df):
    """Crea visualizzazioni per l'analisi del bias"""
    
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Analisi del Bias nel Dataset Adult Income', fontsize=16, fontweight='bold')
    
    # 1. Distribuzione per genere
    gender_income = df.groupby('sex')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    axes[0,0].bar(gender_income.index, gender_income.values, color=['lightcoral', 'lightblue'])
    axes[0,0].set_title('% Reddito >50K per Genere')
    axes[0,0].set_ylabel('Percentuale (%)')
    for i, v in enumerate(gender_income.values):
        axes[0,0].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontweight='bold')
    
    # 2. Distribuzione per etnia
    race_income = df.groupby('race')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    axes[0,1].bar(range(len(race_income)), race_income.values, color='lightgreen')
    axes[0,1].set_title('% Reddito >50K per Etnia')
    axes[0,1].set_ylabel('Percentuale (%)')
    axes[0,1].set_xticks(range(len(race_income)))
    axes[0,1].set_xticklabels(race_income.index, rotation=45, ha='right')
    for i, v in enumerate(race_income.values):
        axes[0,1].text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    
    # 3. Distribuzione per età
    age_income = df.groupby('age_group')['income'].apply(lambda x: (x == '>50K').mean() * 100)
    axes[1,0].bar(range(len(age_income)), age_income.values, color='orange')
    axes[1,0].set_title('% Reddito >50K per Fascia d\'Età')
    axes[1,0].set_ylabel('Percentuale (%)')
    axes[1,0].set_xticks(range(len(age_income)))
    axes[1,0].set_xticklabels(age_income.index)
    for i, v in enumerate(age_income.values):
        axes[1,0].text(i, v + 0.5, f'{v:.1f}%', ha='center')
    
    # 4. Heatmap bias intersezionale (genere x etnia)
    bias_matrix = df.groupby(['sex', 'race'])['income'].apply(lambda x: (x == '>50K').mean() * 100).unstack()
    sns.heatmap(bias_matrix, annot=True, fmt='.1f', cmap='RdYlBu_r', ax=axes[1,1])
    axes[1,1].set_title('% Reddito >50K (Genere x Etnia)')
    axes[1,1].set_xlabel('Etnia')
    axes[1,1].set_ylabel('Genere')
    
    plt.tight_layout()
    plt.show()

def main():
    """Funzione principale per eseguire l'analisi completa"""
    
    # Carica il dataset
    print("Caricamento dataset Adult Income...")
    df = load_and_prepare_data('adult.data.csv')
    
    print(f"Dataset caricato: {len(df):,} righe, {len(df.columns)} colonne")
    print(f"Target: {df['income'].value_counts()}")
    
    # Esegui le analisi
    analyze_group_representation(df)
    analyze_income_bias(df)
    identify_bias_issues(df)
    
    # Crea visualizzazioni
    print("\nGenerazione grafici...")
    create_visualizations(df)
    
    print("\n" + "="*60)
    print("ANALISI COMPLETATA")
    print("="*60)
    print("Il dataset presenta chiari bias che potrebbero influenzare un modello ML:")
    print("- Sovrarappresentazione di uomini bianchi")
    print("- Significative disparità nei tassi di reddito alto tra gruppi")
    print("- Necessarie tecniche di bias mitigation per un ML etico")

if __name__ == "__main__":
    main()