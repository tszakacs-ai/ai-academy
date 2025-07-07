# RISPOSTE ALLE DOMANDE DI DISCUSSIONE
## Analisi del Bias nel Dataset Adult Income

---

### 1. **Quanti dati ci sono per ciascun gruppo (es. uomini/donne, diverse etnie)?**

**Distribuzione per Genere:**
- **Uomini**: ~21.790 esempi (67%)
- **Donne**: ~10.770 esempi (33%)

**Distribuzione per Etnia:**
- **White**: ~27.816 esempi (85.4%)
- **Black**: ~3.124 esempi (9.6%)
- **Asian-Pac-Islander**: ~1.039 esempi (3.2%)
- **Amer-Indian-Eskimo**: ~311 esempi (1.0%)
- **Other**: ~271 esempi (0.8%)

**Problema evidente**: Forte squilibrio con sovrarappresentazione di uomini bianchi.

---

### 2. **Noti differenze marcate nelle percentuali di income alto (>50K) tra i gruppi?**

**Bias di Genere:**
- **Uomini**: ~30.5% ha reddito >50K
- **Donne**: ~11.0% ha reddito >50K
- **Gap**: 19.5 punti percentuali
- **Bias Ratio**: 2.8x (gli uomini hanno quasi 3 volte più probabilità)

**Bias Etnico:**
- **Asian-Pac-Islander**: ~26.8% 
- **White**: ~25.4%
- **Other**: ~22.5%
- **Black**: ~11.7%
- **Amer-Indian-Eskimo**: ~11.6%
- **Gap**: 15.2 punti percentuali tra massimo e minimo

**Bias per Età:**
- **18-25**: ~6.1%
- **26-35**: ~20.2%
- **36-45**: ~31.4%
- **46-55**: ~34.1%
- **56+**: ~24.8%

---

### 3. **Secondo te, queste differenze riflettono la realtà o potrebbero essere dovute a come sono stati raccolti i dati?**

Le differenze osservate sono probabilmente una **combinazione di entrambi i fattori**:

**Possibili riflessi della realtà (anni '90 USA):**
- Gap salariale di genere storicamente documentato
- Discriminazione sistemica nel mercato del lavoro
- Differenze nell'accesso all'istruzione superiore
- Barriere culturali e sociali per le donne nel mondo del lavoro

**Possibili bias nella raccolta dati:**
- **Campionamento non rappresentativo**: sovrarappresentazione di certi gruppi demografici
- **Bias di selezione**: possibile focus su determinate aree geografiche o settori
- **Bias temporale**: i dati riflettono un'epoca specifica (Census 1994)
- **Definizioni categoriche semplificate**: classificazioni etniche limitate

**Conclusione**: Mentre alcune disparità riflettono reali disuguaglianze storiche, la severità degli squilibri suggerisce anche problemi metodologici nella raccolta.

---

### 4. **Se un modello si addestra su questi dati, quali rischi di bias vedi?**

**Rischi Maggiori:**

**A) Bias di Predizione:**
- Il modello imparerà che "essere maschio" è un forte predittore di reddito alto
- Sistematicamente sottovaluterà le capacità reddituali delle donne
- Perpetuerà stereotipi etnici nei risultati

**B) Bias di Rappresentazione:**
- Performance scadente su gruppi minoritari (pochi esempi di training)
- Overfitting sui pattern della popolazione dominante (uomini bianchi)
- Impossibilità di generalizzare su nuove popolazioni

**C) Bias di Allocazione:**
- Decisioni discriminatorie in applicazioni reali (prestiti, assunzioni)
- Rinforzo delle disuguaglianze esistenti
- Negazione di opportunità a gruppi già svantaggiati

---

### 5. **Ci sono gruppi sottorappresentati o assenti del tutto? Che impatto può avere?**

**Gruppi Sottorappresentati:**
- **Donne**: solo 33% del dataset
- **Minoranze etniche**: Asian-Pac-Islander (3.2%), Amer-Indian-Eskimo (1.0%), Other (0.8%)
- **Popolazioni non-USA**: molto limitate nelle country di origine

**Gruppi Potenzialmente Assenti:**
- Identità di genere non-binarie
- Nuove categorie etniche/migratorie
- Popolazioni rurali vs urbane (non specificate)

**Impatti:**
- **Erasure statistica**: gruppi ignorati dalle predizioni
- **Performance degradata**: alta varianza/errori sui gruppi minoritari
- **Bias amplificazione**: il modello "dimentica" che esistono questi gruppi
- **Ingiustizia procedurale**: esclusione sistematica dai benefici dell'AI

---

### 6. **Come potresti rendere il dataset più "fair"?**

**Strategie Pre-processing:**

**A) Bilanciamento dei Dati:**
```python
# Re-sampling
- Undersampling della maggioranza (uomini, bianchi)
- Oversampling delle minoranze (SMOTE, ADASYN)
- Stratified sampling per mantenere proporzioni

# Re-weighting
- Pesi inversamente proporzionali alla frequenza del gruppo
- Cost-sensitive learning
```

**B) Arricchimento del Dataset:**
- Raccogliere più dati per gruppi sottorappresentati
- Synthetic data generation per minoranze
- Data augmentation mirata

**C) Feature Engineering:**
- Rimuovere feature palesemente discriminatorie
- Creare feature proxy-aware
- Normalizzazione demografica

---

### 7. **Pensi che il modello AI riuscirebbe a essere imparziale partendo da questi dati?**

**Risposta: NO, molto difficilmente.**

**Problemi Strutturali:**
- I bias sono troppo profondi e sistemici nel dataset
- Il modello non può "inventare" equità che non esiste nei dati
- Pattern discriminatori verrebbero appresi come "verità"

**Limiti dell'Apprendimento:**
- "Garbage in, garbage out": dati biased → modello biased
- Impossibile distinguere correlazioni legittime da bias storici
- Mancanza di controfattuali per gruppi sottorappresentati

**Conclusione**: Servono interventi attivi di bias mitigation, non solo speranza nell'imparzialità automatica.

---

### 8. **Quali soluzioni pratiche proporresti per mitigare il bias nei dati o nel modello?**

**Soluzioni Multi-livello:**

**A) Pre-processing (Dati):**
```python
# Bilanciamento demografico
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_sample_weight

# Re-sampling stratificato
stratified_sample = df.groupby(['sex', 'race']).apply(
    lambda x: x.sample(min_samples_per_group)
)

# Pesi personalizzati
sample_weights = compute_sample_weight(
    class_weight='balanced',
    y=df[['sex', 'race', 'income']]
)
```

**B) In-processing (Modello):**
```python
# Fairness constraints
from aif360.algorithms.inprocessing import FairClassifier

# Multi-task learning con fairness objectives
loss = classification_loss + λ * fairness_penalty

# Adversarial debiasing
model = AdversarialDebiasing(protected_attributes=['sex', 'race'])
```

**C) Post-processing (Output):**
```python
# Threshold optimization per gruppo
from aif360.algorithms.postprocessing import EqOddsPostprocessing

# Calibrazione separata
for group in protected_groups:
    calibrator = CalibratedClassifierCV()
    calibrator.fit(X_group, y_group)
```

**D) Valutazione Continua:**
```python
# Metriche fairness
- Demographic Parity: P(Ŷ=1|A=0) = P(Ŷ=1|A=1)
- Equalized Odds: P(Ŷ=1|Y=1,A=0) = P(Ŷ=1|Y=1,A=1)
- Calibration: P(Y=1|Ŷ=p,A=0) = P(Y=1|Ŷ=p,A=1)

# Monitoring in produzione
fairness_dashboard = create_fairness_monitoring()
```

**E) Governance e Processo:**
- Team diversificato per bias review
- Audit regolari con stakeholder affected
- Documentazione trasparente dei trade-off
- Feedback loop con comunità impattate

**Raccomandazione finale**: Combinare multiple tecniche piuttosto che affidarsi a una sola soluzione "silver bullet".