import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.svm import SVC

# 1. VERİ ÜRETİMİ
# İkili Sınıflandırma için 2 merkezli veri
X_binary, y_binary = make_blobs(n_samples=100, centers=2, random_state=42, cluster_std=2.5)

# Çoklu Sınıflandırma için 3 merkezli veri
X_multi, y_multi = make_blobs(n_samples=150, centers=3, random_state=42, cluster_std=2.5)

# 2. MODEL EĞİTİMİ (Karar sınırlarını çizmek için Lineer SVM kullanıyoruz)
clf_binary = SVC(kernel='linear', C=1.0).fit(X_binary, y_binary)
clf_multi = SVC(kernel='linear', C=1.0).fit(X_multi, y_multi)

# 3. GÖRSELLEŞTİRME FONKSİYONU
def plot_boundaries(ax, clf, X, y, title):
    # Arka plan için ızgara (meshgrid) oluştur
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # Tüm ızgara noktaları için tahmin yap
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Sınırları ve bölgeleri çiz
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xticks([]); ax.set_yticks([])

# 4. ÇİZİM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

plot_boundaries(ax1, clf_binary, X_binary, y_binary, "İkili Sınıflandırma\n(Tek Bir Karar Sınırı)")
plot_boundaries(ax2, clf_multi, X_multi, y_multi, "Çoklu Sınıflandırma\n(Birden Fazla Karar Sınırı)")

plt.tight_layout()
plt.savefig('binary_vs_multiclass.png', dpi=300)
plt.show()