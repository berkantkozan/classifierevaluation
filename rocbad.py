import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import numpy as np

# 1. ADIM: Veri ve Model Hazırlığı (Normal bir model eğitelim)
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Normal olasılıklar (AUC > 0.5 çıkar)
y_prob_normal = model.predict_proba(X_test)[:, 1]

# 2. ADIM: "Ters" Model Simülasyonu (AUC < 0.5 yapmak için)
# Olasılıkları ters çeviriyoruz: p yerine (1-p) kullanıyoruz.
# Bu, modelin pozitif dediğine negatif, negatif dediğine pozitif demesi gibidir.
y_prob_ters = 1 - y_prob_normal

# Ters modelin ROC ve AUC değerlerini hesaplayalım
fpr, tpr, _ = roc_curve(y_test, y_prob_ters)
roc_auc_ters = auc(fpr, tpr)

# 3. ADIM: Grafiği Çizme
plt.figure(figsize=(8, 6))

# Rastgele Tahmin Çizgisi (Referans)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin (AUC = 0.5)')

# Ters Modelin Eğrisi (Köşegenin Altında Kalacak)
plt.plot(fpr, tpr, color='red', lw=3, label=f'Ters Model (AUC = {roc_auc_ters:.2f})')

# 4. ADIM: Ayarlar ve Etiketler
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12)
plt.ylabel('True Positive Rate (Duyarlılık)', fontsize=12)
plt.title('ROC Eğrisi: AUC < 0.5 Durumu', fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

# Durumu açıklayan bir metin ekleyelim
plt.text(0.5, 0.2, 'Kötü Performans Bölgesi\n(Ters Sınıflandırma)',
         fontsize=12, color='red', ha='center', bbox=dict(facecolor='white', alpha=0.8))

# 5. ADIM: Kaydetme
plt.tight_layout()
plt.savefig('auc_0_5_alti_roc.png', dpi=300)
plt.show()

print("AUC < 0.5 grafiği başarıyla üretildi.")
print(f"Ters modelin AUC değeri: {roc_auc_ters:.2f}") # Konsolda 0.5'ten küçük olduğunu teyit edin