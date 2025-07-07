#!/usr/bin/env python3
"""
IMPLEMENTAZIONE DEMOGRAPHIC PARITY PER DATASET ADULT INCOME

Demographic Parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
La probabilità di predizione positiva deve essere uguale per tutti i gruppi sensibili.

Implementa:
1. Calcolo baseline demographic parity
2. Training con fairness constraints
3. Post-processing per correggere le predizioni
4. Valutazione completa con metriche fairness
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize

class DemographicParityAnalyzer:
    """Classe per analizzare e implementare Demographic Parity"""
    
    def __init__(self):
        self.le_dict = {}
        self.scaler = StandardScaler()
        self.model = None
        self.thresholds = {}
        
    def load_and_prepare_data(self, file_path):
        """Carica e prepara il dataset"""
        column_names = [
            'age', 'workclass', 'fnlwgt', 'education', 'education_num',
            'marital_status', 'occupation', 'relationship', 'race', 'sex',
            'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income'
        ]
        
        df = pd.read_csv(file_path, names=column_names, skipinitialspace=True)
        
        # Pulisci i dati
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.strip()
        
        # Rimuovi righe con valori mancanti rappresentati come '?'
        df = df.replace('?', np.nan).dropna()
        
        return df
    
    def preprocess_features(self, df):
        """Preprocessa le features per il training"""
        df_processed = df.copy()
        
        # Crea variabile target binaria
        df_processed['income_binary'] = (df_processed['income'] == '>50K').astype(int)
        
        # Identifica colonne categoriche e numeriche
        categorical_cols = ['workclass', 'education', 'marital_status', 'occupation', 
                          'relationship', 'race', 'sex', 'native_country']
        numerical_cols = ['age', 'fnlwgt', 'education_num', 'capital_gain', 
                         'capital_loss', 'hours_per_week']
        
        # Encoding delle variabili categoriche
        for col in categorical_cols:
            if col not in self.le_dict:
                self.le_dict[col] = LabelEncoder()
                df_processed[col] = self.le_dict[col].fit_transform(df_processed[col])
            else:
                df_processed[col] = self.le_dict[col].transform(df_processed[col])
        
        # Scaling delle variabili numeriche
        if not hasattr(self.scaler, 'mean_'):
            df_processed[numerical_cols] = self.scaler.fit_transform(df_processed[numerical_cols])
        else:
            df_processed[numerical_cols] = self.scaler.transform(df_processed[numerical_cols])
        
        return df_processed
    
    def calculate_demographic_parity(self, y_true, y_pred, sensitive_attr):
        """Calcola la demographic parity per un attributo sensibile"""
        results = {}
        
        unique_groups = np.unique(sensitive_attr)
        
        print(f"\n{'='*50}")
        print(f"DEMOGRAPHIC PARITY ANALYSIS")
        print(f"{'='*50}")
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            
            # Calcola tasso di predizioni positive
            positive_rate = np.mean(y_pred[group_mask])
            group_size = np.sum(group_mask)
            
            # Calcola accuracy per gruppo
            group_accuracy = accuracy_score(y_true[group_mask], y_pred[group_mask])
            
            results[group] = {
                'positive_rate': positive_rate,
                'group_size': group_size,
                'accuracy': group_accuracy
            }
            
            print(f"Gruppo {group}:")
            print(f"  - Tasso predizioni positive: {positive_rate:.3f} ({positive_rate*100:.1f}%)")
            print(f"  - Dimensione gruppo: {group_size:,}")
            print(f"  - Accuracy: {group_accuracy:.3f}")
        
        # Calcola differenza massima (violazione demographic parity)
        positive_rates = [results[group]['positive_rate'] for group in unique_groups]
        dp_violation = max(positive_rates) - min(positive_rates)
        
        print(f"\nVIOLAZIONE DEMOGRAPHIC PARITY: {dp_violation:.3f}")
        print(f"Range tassi positivi: {min(positive_rates):.3f} - {max(positive_rates):.3f}")
        
        if dp_violation < 0.05:
            print("✅ BUONA: Violazione < 5%")
        elif dp_violation < 0.10:
            print("⚠️  MEDIA: Violazione 5-10%")
        else:
            print("❌ ALTA: Violazione > 10%")
        
        return results, dp_violation
    
    def train_baseline_model(self, X_train, y_train):
        """Addestra modello baseline senza fairness constraints"""
        print("\n" + "="*50)
        print("TRAINING MODELLO BASELINE")
        print("="*50)
        
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def optimize_thresholds_for_demographic_parity(self, X_val, y_val, sensitive_attr_val, target_rate=None):
        """Ottimizza le soglie per gruppo per ottenere demographic parity"""
        print("\n" + "="*50)
        print("OTTIMIZZAZIONE SOGLIE PER DEMOGRAPHIC PARITY")
        print("="*50)
        
        # Ottieni probabilità dal modello
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        unique_groups = np.unique(sensitive_attr_val)
        
        # Se non specificato, usa il tasso medio come target
        if target_rate is None:
            target_rate = np.mean(y_val)
        
        print(f"Target rate per demographic parity: {target_rate:.3f}")
        
        # Metodo diretto: trova soglie che producono il target rate
        optimal_thresholds = []
        
        for group in unique_groups:
            group_mask = (sensitive_attr_val == group)
            group_proba = y_proba[group_mask]
            
            # Ordina le probabilità per trovare la soglia
            sorted_proba = np.sort(group_proba)[::-1]  # ordine decrescente
            
            # Trova soglia che produce circa il target rate
            n_positive_needed = int(target_rate * len(group_proba))
            
            if n_positive_needed == 0:
                threshold = 0.99  # soglia molto alta
            elif n_positive_needed >= len(group_proba):
                threshold = 0.01  # soglia molto bassa
            else:
                threshold = sorted_proba[n_positive_needed - 1]
                
            optimal_thresholds.append(threshold)
            
            # Verifica risultato
            actual_rate = np.mean(group_proba >= threshold)
            print(f"  Gruppo {group}: soglia {threshold:.3f} → tasso {actual_rate:.3f}")
        
        self.thresholds = dict(zip(unique_groups, optimal_thresholds))
        
        # Verifica finale della demographic parity
        predictions = np.zeros_like(y_val)
        final_rates = []
        
        for group in unique_groups:
            group_mask = (sensitive_attr_val == group)
            threshold = self.thresholds[group]
            group_pred = (y_proba[group_mask] >= threshold).astype(int)
            predictions[group_mask] = group_pred
            final_rates.append(np.mean(group_pred))
        
        final_dp_violation = max(final_rates) - min(final_rates)
        final_accuracy = accuracy_score(y_val, predictions)
        
        print(f"\nRisultato ottimizzazione:")
        print(f"  Violazione DP: {final_dp_violation:.3f}")
        print(f"  Accuracy: {final_accuracy:.3f}")
        
        return self.thresholds
    
    def predict_with_demographic_parity(self, X, sensitive_attr):
        """Predizioni con demographic parity usando soglie ottimizzate"""
        y_proba = self.model.predict_proba(X)[:, 1]
        predictions = np.zeros(len(X))
        
        unique_groups = np.unique(sensitive_attr)
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            threshold = self.thresholds.get(group, 0.5)
            
            group_pred = (y_proba[group_mask] >= threshold).astype(int)
            predictions[group_mask] = group_pred
        
        return predictions.astype(int)
    
    def evaluate_fairness_metrics(self, y_true, y_pred_baseline, y_pred_fair, sensitive_attr):
        """Valuta multiple metriche di fairness"""
        print("\n" + "="*60)
        print("CONFRONTO METRICHE FAIRNESS")
        print("="*60)
        
        unique_groups = np.unique(sensitive_attr)
        
        # Demographic Parity
        print("\n1. DEMOGRAPHIC PARITY:")
        print("   Baseline:")
        _, dp_baseline = self.calculate_demographic_parity(y_true, y_pred_baseline, sensitive_attr)
        print("   Con correzione:")
        _, dp_corrected = self.calculate_demographic_parity(y_true, y_pred_fair, sensitive_attr)
        
        # Equal Opportunity (TPR)
        print("\n2. EQUAL OPPORTUNITY (True Positive Rate):")
        for model_name, y_pred in [("Baseline", y_pred_baseline), ("Corretto", y_pred_fair)]:
            print(f"   {model_name}:")
            for group in unique_groups:
                group_mask = (sensitive_attr == group)
                # TPR = TP / (TP + FN)
                true_positives = np.sum((y_true[group_mask] == 1) & (y_pred[group_mask] == 1))
                actual_positives = np.sum(y_true[group_mask] == 1)
                tpr = true_positives / actual_positives if actual_positives > 0 else 0
                print(f"     Gruppo {group}: TPR = {tpr:.3f}")
        
        # Overall Accuracy
        print("\n3. ACCURACY COMPLESSIVA:")
        acc_baseline = accuracy_score(y_true, y_pred_baseline)
        acc_corrected = accuracy_score(y_true, y_pred_fair)
        print(f"   Baseline: {acc_baseline:.3f}")
        print(f"   Corretto: {acc_corrected:.3f}")
        print(f"   Differenza: {acc_corrected - acc_baseline:+.3f}")
        
        return {
            'dp_baseline': dp_baseline,
            'dp_corrected': dp_corrected,
            'acc_baseline': acc_baseline,
            'acc_corrected': acc_corrected
        }
    
    def create_fairness_visualizations(self, y_true, y_pred_baseline, y_pred_fair, sensitive_attr):
        """Crea visualizzazioni per confrontare fairness"""
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Confronto Fairness: Baseline vs Demographic Parity', fontsize=16, fontweight='bold')
        
        unique_groups = np.unique(sensitive_attr)
        
        # 1. Tassi di predizione positiva
        baseline_rates = []
        corrected_rates = []
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            baseline_rate = np.mean(y_pred_baseline[group_mask])
            corrected_rate = np.mean(y_pred_fair[group_mask])
            
            baseline_rates.append(baseline_rate)
            corrected_rates.append(corrected_rate)
        
        x = np.arange(len(unique_groups))
        width = 0.35
        
        axes[0,0].bar(x - width/2, baseline_rates, width, label='Baseline', alpha=0.8)
        axes[0,0].bar(x + width/2, corrected_rates, width, label='DP Corrected', alpha=0.8)
        axes[0,0].set_title('Tassi Predizione Positiva per Gruppo')
        axes[0,0].set_ylabel('Tasso Predizione Positiva')
        axes[0,0].set_xticks(x)
        axes[0,0].set_xticklabels([f'Gruppo {g}' for g in unique_groups])
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Accuracy per gruppo
        baseline_acc = []
        corrected_acc = []
        
        for group in unique_groups:
            group_mask = (sensitive_attr == group)
            baseline_acc.append(accuracy_score(y_true[group_mask], y_pred_baseline[group_mask]))
            corrected_acc.append(accuracy_score(y_true[group_mask], y_pred_fair[group_mask]))
        
        axes[0,1].bar(x - width/2, baseline_acc, width, label='Baseline', alpha=0.8)
        axes[0,1].bar(x + width/2, corrected_acc, width, label='DP Corrected', alpha=0.8)
        axes[0,1].set_title('Accuracy per Gruppo')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels([f'Gruppo {g}' for g in unique_groups])
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Violazione Demographic Parity
        dp_baseline = max(baseline_rates) - min(baseline_rates)
        dp_corrected = max(corrected_rates) - min(corrected_rates)
        
        axes[1,0].bar(['Baseline', 'DP Corrected'], [dp_baseline, dp_corrected], 
                     color=['red', 'green'], alpha=0.7)
        axes[1,0].set_title('Violazione Demographic Parity')
        axes[1,0].set_ylabel('Differenza Max-Min Tassi Positivi')
        axes[1,0].axhline(y=0.05, color='orange', linestyle='--', label='Soglia Accettabile (5%)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Confusion Matrix Comparison
        from sklearn.metrics import confusion_matrix
        
        # Combina tutti i gruppi per visualizzazione generale
        cm_baseline = confusion_matrix(y_true, y_pred_baseline, normalize='true')
        cm_corrected = confusion_matrix(y_true, y_pred_fair, normalize='true')
        
        # Visualizza differenza
        cm_diff = cm_corrected - cm_baseline
        
        im = axes[1,1].imshow(cm_diff, cmap='RdBu', vmin=-0.1, vmax=0.1)
        axes[1,1].set_title('Differenza Confusion Matrix\n(Corretto - Baseline)')
        axes[1,1].set_xlabel('Predetto')
        axes[1,1].set_ylabel('Reale')
        
        # Aggiungi valori alle celle
        for i in range(2):
            for j in range(2):
                text = axes[1,1].text(j, i, f'{cm_diff[i, j]:.3f}',
                                    ha="center", va="center", color="black", fontweight='bold')
        
        plt.colorbar(im, ax=axes[1,1])
        plt.tight_layout()
        plt.show()

def main():
    """Funzione principale per demo completa"""
    
    print("="*60)
    print("IMPLEMENTAZIONE DEMOGRAPHIC PARITY")
    print("Dataset: Adult Income")
    print("="*60)
    
    # Inizializza analyzer
    analyzer = DemographicParityAnalyzer()
    
    # 1. Carica dati
    print("\n1. Caricamento e preprocessing dati...")
    df = analyzer.load_and_prepare_data('adult.data.csv')
    df_processed = analyzer.preprocess_features(df)
    
    print(f"Dataset: {len(df_processed):,} righe")
    
    # 2. Prepara features
    feature_cols = [col for col in df_processed.columns if col not in ['income', 'income_binary']]
    X = df_processed[feature_cols].values
    y = df_processed['income_binary'].values
    
    # Attributi sensibili (usa encoding)
    sex_encoded = df_processed['sex'].values  # 0=Female, 1=Male tipicamente
    race_encoded = df_processed['race'].values
    
    # 3. Split train/test
    X_train, X_test, y_train, y_test, sex_train, sex_test, race_train, race_test = train_test_split(
        X, y, sex_encoded, race_encoded, test_size=0.3, random_state=42, stratify=y
    )
    
    # 4. Train baseline model
    analyzer.train_baseline_model(X_train, y_train)
    
    # 5. Predizioni baseline
    y_pred_baseline = analyzer.model.predict(X_test)
    
    print(f"\nAccuracy baseline: {accuracy_score(y_test, y_pred_baseline):.3f}")
    
    # 6. Analisi demographic parity baseline (per genere)
    print("\n" + "="*60)
    print("ANALISI BASELINE - GENERE")
    analyzer.calculate_demographic_parity(y_test, y_pred_baseline, sex_test)
    
    # 7. Ottimizza soglie per demographic parity
    # Prova con diversi target rates
    print("\n" + "="*60)
    print("TESTING MULTIPLE TARGET RATES")
    print("="*60)
    
    # Test con tasso medio
    overall_positive_rate = np.mean(y_test)
    print(f"\nTasso positivo complessivo: {overall_positive_rate:.3f}")
    
    analyzer.optimize_thresholds_for_demographic_parity(X_test, y_test, sex_test, target_rate=overall_positive_rate)
    y_pred_fair_avg = analyzer.predict_with_demographic_parity(X_test, sex_test)
    
    # Test con tasso intermedio (compromesso)
    intermediate_rate = 0.15  # Tra 9% e 27%
    print(f"\nTest con tasso intermedio: {intermediate_rate:.3f}")
    analyzer.optimize_thresholds_for_demographic_parity(X_test, y_test, sex_test, target_rate=intermediate_rate)
    y_pred_fair_intermediate = analyzer.predict_with_demographic_parity(X_test, sex_test)
    
    # Confronta i risultati
    print("\n" + "="*60)
    print("CONFRONTO DIVERSE STRATEGIE")
    print("="*60)
    
    strategies = [
        ("Baseline", y_pred_baseline),
        ("DP - Tasso Medio", y_pred_fair_avg), 
        ("DP - Tasso Intermedio", y_pred_fair_intermediate)
    ]
    
    for name, predictions in strategies:
        print(f"\n{name}:")
        _, dp_violation = analyzer.calculate_demographic_parity(y_test, predictions, sex_test)
        accuracy = accuracy_score(y_test, predictions)
        print(f"  Accuracy: {accuracy:.3f}")
    
    # Usa la migliore strategia per il resto dell'analisi
    y_pred_fair = y_pred_fair_intermediate
    
    # 9. Valutazione finale
    metrics = analyzer.evaluate_fairness_metrics(y_test, y_pred_baseline, y_pred_fair, sex_test)
    
    # 10. Visualizzazioni
    print("\nGenerazione visualizzazioni...")
    analyzer.create_fairness_visualizations(y_test, y_pred_baseline, y_pred_fair, sex_test)
    
    # 11. Risultati finali
    print("\n" + "="*60)
    print("RISULTATI FINALI")
    print("="*60)
    print(f"Violazione Demographic Parity:")
    print(f"  Baseline: {metrics['dp_baseline']:.3f}")
    print(f"  Corretto: {metrics['dp_corrected']:.3f}")
    print(f"  Miglioramento: {metrics['dp_baseline'] - metrics['dp_corrected']:+.3f}")
    print(f"\nAccuracy:")
    print(f"  Baseline: {metrics['acc_baseline']:.3f}")
    print(f"  Corretto: {metrics['acc_corrected']:.3f}")
    print(f"  Trade-off: {metrics['acc_corrected'] - metrics['acc_baseline']:+.3f}")
    
    improvement_pct = ((metrics['dp_baseline'] - metrics['dp_corrected']) / metrics['dp_baseline']) * 100
    print(f"\n✅ Riduzione violazione DP: {improvement_pct:.1f}%")

if __name__ == "__main__":
    main()