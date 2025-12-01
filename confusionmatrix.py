import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. ADIM: MATRİSİ ELLE OLUŞTURMA
# Matris Yapısı: [[TN, FP], [FN, TP]] (Scikit-learn standardı: Üst satır Negatif, Alt satır Pozitif)
# Senaryodaki değerler: TN=1, FP=9, FN=10, TP=90
cm = np.array([[1, 9], 
               [10, 90]])

# 2. ADIM: ETİKETLER VE AÇIKLAMALAR
group_names = ['TN\n(Doğru Negatif)', 'FP\n(Yanlış Pozitif)', 
               'FN\n(Yanlış Negatif)', 'TP\n(Doğru Pozitif)']
group_counts = [f"{value:.0f}" for value in cm.flatten()]
# Yüzdeleri de hesaplayalım (Satır bazlı)
group_percentages = [f"({value/cm.sum(axis=1)[i//2]:.1%})" for i, value in enumerate(cm.flatten())]

# Kutuların içine yazılacak metni birleştirelim
labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
labels = np.asarray(labels).reshape(2,2)

# 3. ADIM: GRAFİĞİ ÇİZME
plt.figure(figsize=(8, 6))
# 'Blues' renk haritası kullanıyoruz. 'annot=labels' ile hazırladığımız metinleri kutulara yazıyoruz.
sns.heatmap(cm, annot=labels, fmt='', cmap='Blues', cbar=False,
            annot_kws={"size": 14, "fontweight": "bold"}, linewidths=2, linecolor='black')

# 4. ADIM: EKSEN VE BAŞLIK AYARLARI
plt.title('Arızalı Ürünleri Kaçıran Modelin Karmaşıklık Matrisi\n(F1 Yüksek, MCC Düşük Senaryosu)', fontsize=14, pad=20)
plt.xlabel('Tahmin Edilen Sınıf', fontsize=12, labelpad=10)
plt.ylabel('Gerçek Sınıf', fontsize=12, labelpad=10)

# Eksen etiketlerini değiştirelim (0: Arızalı, 1: Sağlam)
plt.xticks([0.5, 1.5], ['ARIZALI (Negatif-0)', 'SAĞLAM (Pozitif-1)'], fontsize=11)
plt.yticks([0.5, 1.5], ['ARIZALI (Negatif-0)', 'SAĞLAM (Pozitif-1)'], fontsize=11, rotation=0)

# Kritik hatayı vurgulamak için bir kutu çizelim (FP kutusu)
import matplotlib.patches as patches
ax = plt.gca()
# (1, 0) koordinatındaki kutu (FP: 9)
rect = patches.Rectangle((1, 0), 1, 1, linewidth=4, edgecolor='red', facecolor='none')
ax.add_patch(rect)
plt.text(1.5, 0.25, "KRİTİK HATA!\n9 Arızalı ürüne\n'Sağlam' dendi.", 
         color='red', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('sekil_12_confusion_matrix_scenario.png', dpi=300)
plt.show()