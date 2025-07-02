import csv
import random

random.seed(42)
from collections import Counter, defaultdict
from math import log

DATA_PATH = "Giorno_12/spam.csv"


def preprocess(text: str) -> list[str]:
    """Lowercase and split text into tokens."""
    return text.lower().split()


def load_data(path: str) -> list[tuple[str, str]]:
    data = []
    with open(path, encoding="latin-1") as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            if len(row) >= 2:
                label, text = row[0], row[1]
                data.append((label, text))
    random.shuffle(data)
    return data


def train_test_split(data: list[tuple[str, str]], test_ratio: float = 0.2):
    split = int(len(data) * (1 - test_ratio))
    return data[:split], data[split:]


class NaiveBayes:
    def __init__(self):
        self.label_counts = Counter()
        self.word_counts = defaultdict(Counter)
        self.vocab = set()

    def fit(self, data: list[tuple[str, str]]):
        for label, text in data:
            self.label_counts[label] += 1
            words = preprocess(text)
            for w in words:
                self.word_counts[label][w] += 1
                self.vocab.add(w)

    def predict(self, text: str) -> str:
        words = preprocess(text)
        best_label = None
        best_log_prob = float("-inf")
        n_docs = sum(self.label_counts.values())
        vocab_size = len(self.vocab)
        for label in self.label_counts:
            log_prob = log(self.label_counts[label] / n_docs)
            total_words = sum(self.word_counts[label].values())
            for w in words:
                count = self.word_counts[label][w] + 1
                log_prob += log(count / (total_words + vocab_size))
            if log_prob > best_log_prob:
                best_log_prob = log_prob
                best_label = label
        return best_label


def evaluate(model: NaiveBayes, data: list[tuple[str, str]]):
    tp = fp = tn = fn = 0
    for label, text in data:
        pred = model.predict(text)
        if label == "spam" and pred == "spam":
            tp += 1
        elif label == "spam" and pred == "ham":
            fn += 1
        elif label == "ham" and pred == "spam":
            fp += 1
        else:
            tn += 1
    accuracy = (tp + tn) / len(data)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "confusion": [[tp, fp], [fn, tn]],
    }


def print_risk_analysis():
    print("\n--- EU AI Act Risk Analysis (simplified) ---")
    print("* False positives: legitimate messages flagged as spam -> potential loss of important communication")
    print("* False negatives: spam that bypasses the filter -> risk of phishing or fraud")
    print("* Transparency: document dataset source, preprocessing, metrics, model version")
    print("* Audit: log false positive/negative rates over time and review samples periodically")
    print("* Documentation: describe data cleaning, evaluation metrics, limitations (language coverage, message length, etc.)")


def main():
    data = load_data(DATA_PATH)
    train_data, test_data = train_test_split(data)
    model = NaiveBayes()
    model.fit(train_data)
    metrics = evaluate(model, test_data)
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print_risk_analysis()


if __name__ == "__main__":
    main()
