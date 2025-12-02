import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# SENARYO: Model A ve Model B karşılaştırılıyor
# n_00: İkisi de Yanlış (10 adet)
# n_11: İkisi de Doğru (80 adet)
# n_10: Model A Doğru, Model B Yanlış (25 adet) -> Model A daha iyi gibi
# n_01: Model A Yanlış, Model B Doğru (5 adet)

data = np.array([[80, 5], 
                 [25, 10]])

# Etiketler
labels = np.array([['İkisi de Doğru\n(Farksız)', 'Model A Yanlış\nModel B Doğru'],
                   ['Model A Doğru\nModel B Yanlış', 'İkisi de Yanlış\n(Farksız)']])

# Görselleştirme
plt.figure(figsize=(9, 7))
sns.heatmap(data, annot=labels, fmt='', cmap='Purples', cbar=False,
            annot_kws={"size": 13, "fontweight": "bold"}, linewidths=2, linecolor='black')

# Değerleri (Sayıları) ayrıca ekleyelim
for i in range(2):
    for j in range(2):
        plt.text(j+0.5, i+0.7, f"Sayı: {data[i, j]}", 
                 ha='center', va='center', fontsize=14, color='darkred')

plt.title('McNemar Testi için Anlaşmazlık Tablosu (Contingency Table)\nİstatistiki Farkı Belirleyen Alanlar: Sol Alt ve Sağ Üst Köşeler', fontsize=14, pad=20)
plt.xlabel('Model B Performansı', fontsize=12)
plt.ylabel('Model A Performansı', fontsize=12)
plt.xticks([0.5, 1.5], ['Doğru', 'Yanlış'])
plt.yticks([0.5, 1.5], ['Doğru', 'Yanlış'])

# Vurgulama (Fark yaratan kısımlar)
import matplotlib.patches as patches
ax = plt.gca()
# n_10 kutusu (Sol Alt) - Model A'nın üstünlüğü
rect1 = patches.Rectangle((0, 1), 1, 1, linewidth=4, edgecolor='green', facecolor='none')
ax.add_patch(rect1)
# n_01 kutusu (Sağ Üst) - Model B'nin üstünlüğü
rect2 = patches.Rectangle((1, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
ax.add_patch(rect2)

plt.tight_layout()
plt.savefig('mcnemar_tablosu.png', dpi=300)
plt.show()