import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
# Gerekli metrikleri import ediyoruz
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

# 1. ADIM: DENGESİZ VERİ SETİ OLUŞTURMA
# weights=[0.95, 0.05] parametresi ile %95 sınıf 0 (Negatif), %5 sınıf 1 (Pozitif) üretiyoruz.
X, y = make_classification(n_samples=2000, n_features=20, n_classes=2,
                           weights=[0.95, 0.05], random_state=42)

# Veriyi eğitim ve test olarak ayırıyoruz (stratify=y, dengesizliği korur)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# 2. ADIM: MODEL EĞİTİMİ
# Standart bir Lojistik Regresyon modeli eğitiyoruz
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Test seti için olasılık tahminlerini alıyoruz
y_prob = model.predict_proba(X_test)[:, 1]

# --- GRAFİK ÇİZİM HAZIRLIĞI ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 3. ADIM: SOL GRAFİK - ROC EĞRİSİ
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'Model (AUC = {roc_auc:.2f})')
ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin')
ax1.set_xlim([-0.02, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12)
ax1.set_ylabel('True Positive Rate (Duyarlılık)', fontsize=12)
ax1.set_title('ROC Eğrisi \nDengesiz Veride İyimser Görünüm', fontsize=13, fontweight='bold')
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)
# Yorum ekleyelim
ax1.text(0.5, 0.4, "Eğri sol üste yakın duruyor,\nbaşarılı gibi görünüyor.", color='green', ha='center')

# 4. ADIM: SAĞ GRAFİK - PR EĞRİSİ
precision, recall, _ = precision_recall_curve(y_test, y_prob)
# PR eğrisi için Average Precision (AP) skoru kullanılır
pr_auc = average_precision_score(y_test, y_prob)
# Dengesiz veride rastgele tahmin çizgisi (Pozitif sınıf oranı)
baseline_pr = y_test.sum() / len(y_test)

ax2.plot(recall, precision, color='purple', lw=2, label=f'Model (AP = {pr_auc:.2f})')
ax2.axhline(y=baseline_pr, color='gray', lw=2, linestyle='--', label=f'Rastgele Baz ({baseline_pr:.2f})')
ax2.set_xlim([0.0, 1.02])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('Recall (Duyarlılık)', fontsize=12)
ax2.set_ylabel('Precision (Kesinlik)', fontsize=12)
ax2.set_title('PR Eğrisi \nGerçek Performans (Düşük Kesinlik)', fontsize=13, fontweight='bold')
ax2.legend(loc="upper right")
ax2.grid(True, alpha=0.3)
# Yorum ekleyelim
ax2.text(0.5, 0.6, "Duyarlılık arttıkça\nKesinlik hızla düşüyor!", color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8))

# 5. ADIM: KAYDETME VE GÖSTERME
plt.tight_layout()
plt.savefig('sekil_5_roc_vs_pr_dengesiz.png', dpi=300)
plt.show()

print(f"Veri setindeki pozitif sınıf oranı (Baseline PR): {baseline_pr:.3f}")
print("Şekil 5 başarıyla üretildi.")