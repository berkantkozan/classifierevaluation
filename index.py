import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 1. ADIM: Örnek Veri Üretimi (Kendi veriniz varsa burayı atlayın)
# Rastgele bir sınıflandırma problemi yaratıyoruz
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 2. ADIM: Modeli Eğitme (Örnek olarak Random Forest)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Tahminleri alıyoruz
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # ROC için olasılık değerleri lazım

# 3. ADIM: Görselleştirme
# Yan yana iki grafik çizmek için alan oluşturuyoruz
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# --- A. Confusion Matrix (Soldaki Grafik) ---
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1, cbar=False,
            annot_kws={"size": 14}) # Yazı boyutu
ax1.set_title('Confusion Matrix (Karmaşıklık Matrisi)', fontsize=16)
ax1.set_xlabel('Tahmin Edilen Etiket', fontsize=12)
ax1.set_ylabel('Gerçek Etiket', fontsize=12)
ax1.set_xticklabels(['Negatif (0)', 'Pozitif (1)'])
ax1.set_yticklabels(['Negatif (0)', 'Pozitif (1)'])

# --- B. ROC Eğrisi (Sağdaki Grafik) ---
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Eğrisi (AUC = {roc_auc:.2f})')
ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Şans çizgisi
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12)
ax2.set_ylabel('True Positive Rate (Doğru Pozitif Oranı)', fontsize=12)
ax2.set_title('ROC Curve (ROC Eğrisi)', fontsize=16)
ax2.legend(loc="lower right")
ax2.grid(True, alpha=0.3)

# 4. ADIM: Kaydetme
plt.tight_layout()
# dpi=300 akademik baskı kalitesidir.
plt.savefig('siniflandirici_degerlendirme_grafigi.png', dpi=300) 
plt.show()

print("Grafik başarıyla oluşturuldu ve kaydedildi!")