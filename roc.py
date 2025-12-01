import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc

# 1. ADIM: "İyi Model" için sahte veri üretelim
# (Mükemmel ve Rastgele modelleri elle çizeceğiz, bunu sadece "Ara Model" olsun diye yapıyoruz)
X, y = make_classification(n_samples=1000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# 2. ADIM: Grafiği Çizme
plt.figure(figsize=(10, 8))

# A. Mükemmel Sınıflandırıcı (Mavi Çizgi - Köşeli)
# (0,0) -> (0,1) -> (1,1) noktalarını birleştirir
plt.plot([0, 0, 1], [0, 1, 1], color='blue', lw=3, label='Mükemmel Sınıflandırıcı (AUC = 1.0)')

# B. Tipik/İyi Bir Sınıflandırıcı (Turuncu Eğri - Örnek Veriden)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'İyi Bir Sınıflandırıcı (AUC = {roc_auc:.2f})')

# C. Rastgele Tahmin (Kesik Çizgi)
# (0,0) -> (1,1) diyagonal çizgi
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Rastgele Tahmin (AUC = 0.5)')

# 3. ADIM: Ayarlar ve Etiketler
plt.xlim([-0.02, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)', fontsize=12)
plt.ylabel('True Positive Rate (Duyarlılık)', fontsize=12)
plt.title('ROC Eğrisi Performans Karşılaştırması', fontsize=15)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)

# Okuyucunun anlamasını kolaylaştıran oklar ve yazılar (İsteğe bağlı)
plt.text(0.6, 0.2, 'Daha Kötü Performans', fontsize=10, rotation=45, color='gray')
plt.text(0.2, 0.8, 'Daha İyi Performans', fontsize=10, rotation=0, color='blue')

# 4. ADIM: Kaydetme
plt.tight_layout()
plt.savefig('sekil_4_roc_egrisi.png', dpi=300) # 300 DPI baskı kalitesidir
plt.show()

print("Şekil 4 başarıyla üretildi.")