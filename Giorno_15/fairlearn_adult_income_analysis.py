#!/usr/bin/env python3
"""
ESERCIZIO: FAIRLEARN SU ADULT INCOME

Implementazione completa dell'analisi del bias usando Fairlearn di Microsoft.
Misura quanto il modello "amplifica", "riduce" o "trasforma" lo sbilanciamento 
presente nei dati originali.

Fairlearn permette di:
- Confrontare tasso predizioni positive tra gruppi protetti
- Identificare se il modello sta trasferendo (o aumentando) il bias 
- Vedere concretamente che un modello "corretto" può essere poco equo
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Importa Fairlearn
from fairlearn.metrics import MetricFrame
from fairlearn.metrics import selection_rate, demographic_parity_difference, equalized_odds_difference
from fairlearn.postprocessing import ThresholdOptimizer
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import matplotlib.pyplot as plt
import seaborn as sns

class FairlearnAnalyzer:
    """Classe per analisi completa con Fairlearn"""
    
    def __init__(self):
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.baseline_model = None
        self.fair_model = None
        
    def load_and_prepare_data(self, file_path):
        """Carica e prepara il dataset Adult Income"""
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        
        df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)
        
        # Pulisci i dati
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        # Rimuovi righe con valori mancanti
        df = df.replace('?', np.nan).dropna()
        
        # Crea target binario
        df['income_binary'] = (df['income'] == '>50K').astype(int)
        
        return df
    
    def preprocess_features(self, df, fit_encoders=True):
        """Preprocessa features per ML"""
        df_processed = df.copy()
        
        categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                          'relationship', 'race', 'sex', 'native_country']
        numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                         'capital_loss', 'hours_per_week']
        
        # Encoding variabili categoriche
        for col in categorical_cols:
            if fit_encoders:
                if col not in self.le_dict:
                    self.le_dict[col] = LabelEncoder()
                    df_processed[col] = self.le_dict[col].fit_transform(df_processed[col])
                else:
                    df_processed[col] = self.le_dict[col].transform(df_processed[col])
            else:
                df_processed[col] = self.le_dict[col].transform(df_processed[col])
        
        # Scaling variabili numeriche
        if fit_encoders:
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        else:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def analyze_data_bias(self, df):
        """Analizza il bias presente nei dati originali"""
        print("="*70)
        print("ANALISI BIAS NEI DATI ORIGINALI")
        print("="*70)
        
        # Bias di genere nei dati
        print("\n📊 BIAS DI GENERE NEI DATI:")
        gender_income = df.groupby('sex')['income_binary'].agg(['count', 'mean'])
        
        for gender in gender_income.index:
            count = gender_income.loc[gender, 'count']
            rate = gender_income.loc[gender, 'mean']
            print(f"   {gender}: {count:,} persone, {rate:.1%} con reddito >50K")
        
        # Calcola gap nei dati originali
        female_rate = gender_income.loc['Female', 'mean']
        male_rate = gender_income.loc['Male', 'mean']
        data_gap = male_rate - female_rate
        
        print(f"\n📈 GAP NEI DATI ORIGINALI: {data_gap:.1%}")
        print(f"   Rapporto M/F: {male_rate/female_rate:.2f}x")
        
        # Bias etnico nei dati
        print("\n🌍 BIAS ETNICO NEI DATI:")
        race_income = df.groupby('race')['income_binary'].agg(['count', 'mean']).sort_values('mean', ascending=False)
        
        for race in race_income.index:
            count = race_income.loc[race, 'count']
            rate = race_income.loc[race, 'mean']
            print(f"   {race}: {count:,} persone, {rate:.1%} con reddito >50K")
        
        return {
            'gender_gap': data_gap,
            'male_rate': male_rate,
            'female_rate': female_rate
        }
    
    def train_baseline_model(self, X_train, y_train):
        """Addestra modello baseline senza correzioni fairness"""
        print("\n" + "="*70)
        print("TRAINING MODELLO BASELINE (senza fairness)")
        print("="*70)
        
        self.baseline_model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.baseline_model.fit(X_train, y_train)
        
        print("✅ Modello baseline addestrato")
        return self.baseline_model
    
    def analyze_model_bias_with_fairlearn(self, X_test, y_test, sensitive_features, model, model_name="Modello"):
        """Analizza bias del modello usando Fairlearn MetricFrame"""
        print(f"\n📊 ANALISI FAIRNESS - {model_name.upper()}")
        print("="*50)
        
        # Predizioni - gestisce sia modelli normali che ThresholdOptimizer
        if hasattr(model, 'predict') and 'sensitive_features' in model.predict.__code__.co_varnames:
            # Per ThresholdOptimizer che richiede sensitive_features
            y_pred = model.predict(X_test, sensitive_features=sensitive_features)
        else:
            # Per modelli normali
            y_pred = model.predict(X_test)
        
        # Crea MetricFrame per analisi fairness
        mf = MetricFrame(
            metrics={
                'accuracy': accuracy_score,
                'selection_rate': selection_rate,  # Tasso predizioni positive
            },
            y_true=y_test,
            y_pred=y_pred,
            sensitive_features=sensitive_features
        )
        
        print("📈 RISULTATI PER GRUPPO:")
        print(mf.by_group)
        
        # Calcola differenze tra gruppi
        print(f"\n📊 DIFFERENZE TRA GRUPPI:")
        
        # Selection rate difference (Demographic Parity)
        sr_diff = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_features)
        print(f"   Demographic Parity Difference: {sr_diff:.3f}")
        
        # Equalized Odds difference
        eo_diff = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_features)
        print(f"   Equalized Odds Difference: {eo_diff:.3f}")
        
        # Interpretazione
        if abs(sr_diff) < 0.05:
            print("   ✅ Demographic Parity: BUONA (< 5%)")
        elif abs(sr_diff) < 0.10:
            print("   ⚠️  Demographic Parity: MEDIA (5-10%)")
        else:
            print("   ❌ Demographic Parity: VIOLA (> 10%)")
        
        return mf, sr_diff, eo_diff
    
    def compare_data_vs_model_bias(self, data_stats, model_sr_diff):
        """Confronta bias nei dati vs bias nel modello"""
        print("\n" + "="*70)
        print("🔍 CONFRONTO: BIAS DATI vs BIAS MODELLO")
        print("="*70)
        
        data_gap = data_stats['gender_gap']
        model_gap = abs(model_sr_diff)
        
        print(f"📊 Gap nei dati originali: {data_gap:.1%}")
        print(f"🤖 Gap nelle predizioni modello: {model_gap:.1%}")
        
        # Determina se il modello amplifica, riduce o mantiene il bias
        if model_gap > data_gap * 1.1:  # 10% di tolleranza
            print("📈 Il modello STA AMPLIFICANDO il bias presente nei dati")
            effect = "AMPLIFICA"
        elif model_gap < data_gap * 0.9:  # 10% di tolleranza
            print("📉 Il modello STA RIDUCENDO il bias presente nei dati") 
            effect = "RIDUCE"
        else:
            print("➡️  Il modello STA TRASFERENDO il bias dai dati alle predizioni")
            effect = "TRASFERISCE"
        
        amplification_factor = model_gap / data_gap if data_gap > 0 else float('inf')
        print(f"🔢 Fattore di amplificazione: {amplification_factor:.2f}x")
        
        return effect, amplification_factor
    
    def train_fair_model_with_postprocessing(self, X_train, y_train, X_val, y_val, sensitive_train, sensitive_val):
        """Addestra modello fair usando post-processing"""
        print("\n" + "="*70)
        print("TRAINING MODELLO FAIR (con post-processing)")
        print("="*70)
        
        # Usa ThresholdOptimizer per demographic parity
        postprocess_est = ThresholdOptimizer(
            estimator=self.baseline_model,
            constraints="demographic_parity",
            prefit=True  # Il modello è già addestrato
        )
        
        postprocess_est.fit(X_val, y_val, sensitive_features=sensitive_val)
        
        self.fair_model = postprocess_est
        print("✅ Modello fair (post-processing) addestrato")
        
        return self.fair_model
    
    def train_fair_model_with_inprocessing(self, X_train, y_train, sensitive_train):
        """Addestra modello fair usando in-processing"""
        print("\n" + "="*70)
        print("TRAINING MODELLO FAIR (con in-processing)")
        print("="*70)
        
        # Usa ExponentiatedGradient con DemographicParity constraint
        mitigator = ExponentiatedGradient(
            RandomForestClassifier(n_estimators=50, random_state=42),
            constraints=DemographicParity()
        )
        
        mitigator.fit(X_train, y_train, sensitive_features=sensitive_train)
        
        self.fair_model_inprocess = mitigator
        print("✅ Modello fair (in-processing) addestrato")
        
        return self.fair_model_inprocess
    
    def create_comparative_visualizations(self, X_test, y_test, sensitive_features):
        """Crea visualizzazioni comparative"""
        
        # Predizioni di tutti i modelli - gestisce ThresholdOptimizer correttamente
        y_pred_baseline = self.baseline_model.predict(X_test)
        
        # Per ThresholdOptimizer serve sensitive_features
        y_pred_fair_post = self.fair_model.predict(X_test, sensitive_features=sensitive_features)
        
        # Per ExponentiatedGradient predict normale
        y_pred_fair_in = self.fair_model_inprocess.predict(X_test)
        
        # Calcola selection rates
        models = {
            'Dati Originali': y_test,
            'Baseline': y_pred_baseline,
            'Fair (Post)': y_pred_fair_post,
            'Fair (In)': y_pred_fair_in
        }
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confronto Fairness: Dati vs Modelli', fontsize=16, fontweight='bold')
        
        # 1. Selection rates per gruppo
        selection_rates = {}
        for name, predictions in models.items():
            rates = []
            unique_groups = np.unique(sensitive_features)
            
            for group in unique_groups:
                group_mask = (sensitive_features == group)
                rate = np.mean(predictions[group_mask])
                rates.append(rate)
            
            selection_rates[name] = rates
        
        x = np.arange(len(unique_groups))
        width = 0.2
        
        for i, (name, rates) in enumerate(selection_rates.items()):
            axes[0,0].bar(x + i*width, rates, width, label=name, alpha=0.8)
        
        axes[0,0].set_title('Tasso Predizioni Positive per Gruppo')
        axes[0,0].set_ylabel('Selection Rate')
        axes[0,0].set_xticks(x + width*1.5)
        axes[0,0].set_xticklabels(['Female', 'Male'])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Demographic Parity Violations
        dp_violations = []
        model_names = []
        
        for name, predictions in models.items():
            dp_diff = abs(demographic_parity_difference(y_test, predictions, sensitive_features=sensitive_features))
            dp_violations.append(dp_diff)
            model_names.append(name)
        
        colors = ['blue', 'red', 'green', 'orange']
        bars = axes[0,1].bar(model_names, dp_violations, color=colors, alpha=0.7)
        axes[0,1].set_title('Violazione Demographic Parity')
        axes[0,1].set_ylabel('Demographic Parity Difference')
        axes[0,1].axhline(y=0.05, color='orange', linestyle='--', label='Soglia Accettabile (5%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Aggiungi valori sulle barre
        for bar, value in zip(bars, dp_violations):
            axes[0,1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                          f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 3. Accuracy per gruppo
        for i, (name, predictions) in enumerate(list(models.items())[1:], 1):  # Skip dati originali per accuracy
            accuracies = []
            for group in unique_groups:
                group_mask = (sensitive_features == group)
                acc = accuracy_score(y_test[group_mask], predictions[group_mask])
                accuracies.append(acc)
            
            x_pos = np.arange(len(unique_groups)) + (i-1)*width
            axes[1,0].bar(x_pos, accuracies, width, label=name, alpha=0.8)
        
        axes[1,0].set_title('Accuracy per Gruppo')
        axes[1,0].set_ylabel('Accuracy')
        axes[1,0].set_xticks(np.arange(len(unique_groups)) + width)
        axes[1,0].set_xticklabels(['Female', 'Male'])
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Trade-off Accuracy vs Fairness
        overall_accuracies = []
        for name, predictions in list(models.items())[1:]:  # Skip dati originali
            acc = accuracy_score(y_test, predictions)
            overall_accuracies.append(acc)
        
        model_names_no_data = list(models.keys())[1:]
        dp_violations_models = dp_violations[1:]  # Skip dati originali
        
        scatter = axes[1,1].scatter(dp_violations_models, overall_accuracies, 
                                  s=100, alpha=0.7, c=['red', 'green', 'orange'])
        
        axes[1,1].set_title('Trade-off: Accuracy vs Fairness')
        axes[1,1].set_xlabel('Demographic Parity Violation')
        axes[1,1].set_ylabel('Overall Accuracy')
        
        # Annota i punti
        for i, name in enumerate(model_names_no_data):
            axes[1,1].annotate(name, (dp_violations_models[i], overall_accuracies[i]),
                              xytext=(5, 5), textcoords='offset points')
        
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """Funzione principale per l'analisi completa con Fairlearn"""
    
    print("="*70)
    print("🎯 ESERCIZIO: FAIRLEARN SU ADULT INCOME")
    print("Analisi completa del bias con strumenti Microsoft Fairlearn")
    print("="*70)
    
    # Inizializza analyzer
    analyzer = FairlearnAnalyzer()
    
    # 1. Carica e prepara dati
    print("\n1️⃣ Caricamento dati...")
    df = analyzer.load_and_prepare_data('adult.data.csv')
    print(f"Dataset caricato: {len(df):,} righe")
    
    # 2. Analizza bias nei dati originali
    data_stats = analyzer.analyze_data_bias(df)
    
    # 3. Preprocessing per ML
    df_processed = analyzer.preprocess_features(df, fit_encoders=True)
    
    # Prepara features e target
    feature_cols = [col for col in df_processed.columns if col not in ['income', 'income_binary']]
    X = df_processed[feature_cols].values
    y = df_processed['income_binary'].values
    sex_encoded = df_processed['sex'].values  # Attributo sensibile
    
    # 4. Split dei dati
    X_train, X_temp, y_train, y_temp, sex_train, sex_temp = train_test_split(
        X, y, sex_encoded, test_size=0.4, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test, sex_val, sex_test = train_test_split(
        X_temp, y_temp, sex_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"\nSplit dati: {len(X_train):,} train, {len(X_val):,} val, {len(X_test):,} test")
    
    # 5. Addestra modello baseline
    analyzer.train_baseline_model(X_train, y_train)
    
    # 6. Analizza bias modello baseline con Fairlearn
    mf_baseline, sr_diff_baseline, eo_diff_baseline = analyzer.analyze_model_bias_with_fairlearn(
        X_test, y_test, sex_test, analyzer.baseline_model, "BASELINE"
    )
    
    # 7. Confronta bias dati vs modello
    effect, amplification = analyzer.compare_data_vs_model_bias(data_stats, sr_diff_baseline)
    
    # 8. Addestra modelli fair
    analyzer.train_fair_model_with_postprocessing(X_train, y_train, X_val, y_val, sex_train, sex_val)
    analyzer.train_fair_model_with_inprocessing(X_train, y_train, sex_train)
    
    # 9. Analizza modelli fair
    mf_fair_post, sr_diff_post, eo_diff_post = analyzer.analyze_model_bias_with_fairlearn(
        X_test, y_test, sex_test, analyzer.fair_model, "FAIR (POST-PROCESSING)"
    )
    
    mf_fair_in, sr_diff_in, eo_diff_in = analyzer.analyze_model_bias_with_fairlearn(
        X_test, y_test, sex_test, analyzer.fair_model_inprocess, "FAIR (IN-PROCESSING)"
    )
    
    # 10. Risultati finali
    print("\n" + "="*70)
    print("📊 RISULTATI FINALI")
    print("="*70)
    
    print(f"\n🔍 BIAS NEI DATI ORIGINALI:")
    print(f"   Gap di genere: {data_stats['gender_gap']:.1%}")
    
    print(f"\n🤖 BIAS NEI MODELLI:")
    print(f"   Baseline: {abs(sr_diff_baseline):.1%}")
    print(f"   Fair (Post): {abs(sr_diff_post):.1%}")
    print(f"   Fair (In): {abs(sr_diff_in):.1%}")
    
    print(f"\n📈 EFFETTO DEL MODELLO BASELINE:")
    print(f"   Il modello {effect} il bias (fattore {amplification:.2f}x)")
    
    # Calcola miglioramenti
    improvement_post = ((abs(sr_diff_baseline) - abs(sr_diff_post)) / abs(sr_diff_baseline)) * 100
    improvement_in = ((abs(sr_diff_baseline) - abs(sr_diff_in)) / abs(sr_diff_baseline)) * 100
    
    print(f"\n✅ MIGLIORAMENTI FAIRNESS:")
    print(f"   Post-processing: {improvement_post:.1f}% riduzione bias")
    print(f"   In-processing: {improvement_in:.1f}% riduzione bias")
    
    # 11. Visualizzazioni
    print("\n📊 Generazione visualizzazioni...")
    analyzer.create_comparative_visualizations(X_test, y_test, sex_test)
    
    print("\n🎉 ANALISI COMPLETATA!")
    print("📚 Fairlearn ha mostrato come il modello trasforma il bias dei dati")

if __name__ == "__main__":
    main()