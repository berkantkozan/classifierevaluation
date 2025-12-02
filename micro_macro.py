import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import f1_score

# SENARYO: Çoklu Sınıflandırma (3 Sınıf)
# Sınıf 0 (Çoğunluk): 100 örnek (Model hepsini biliyor)
# Sınıf 1 (Azınlık): 10 örnek (Model hepsini yanlış biliyor)
# Sınıf 2 (Azınlık): 10 örnek (Model hepsini yanlış biliyor)

y_true = np.array([0]*100 + [1]*10 + [2]*10)
# Model Sınıf 0'ı süper biliyor, ama Sınıf 1 ve 2 yerine de '0' diyor.
y_pred = np.array([0]*100 + [0]*10 + [0]*10)

# Hesaplama (average parametresi ile)
f1_micro = f1_score(y_true, y_pred, average='micro')
f1_macro = f1_score(y_true, y_pred, average='macro')
# Weighted (Ağırlıklı) da ekleyelim, farkı görmek için
f1_weighted = f1_score(y_true, y_pred, average='weighted')

# Görselleştirme
metrics = ['Micro-F1\n(Genel Performans)', 'Weighted-F1\n(Ağırlıklı)', 'Macro-F1\n(Sınıf Bazlı Eşitlik)']
values = [f1_micro, f1_weighted, f1_macro]
colors = ['#3498db', '#95a5a6', '#e74c3c'] # Macro Kırmızı (Düşük çıkacak)

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=colors, edgecolor='black', width=0.6)

# Değerleri yaz
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=14, fontweight='bold')

plt.title('Çoklu Sınıflandırmada Mikro ve Makro Ortalama Farkı\n(Azınlık Sınıfları Başarısız Olduğunda)', fontsize=14)
plt.ylabel('F1-Skoru', fontsize=12)
plt.ylim(0, 1.1)

# Açıklama Kutusu
plt.text(1.85, 0.5, "Model büyük sınıfı (Sınıf 0)\nbildiği için Micro-F1 (%83)\nyüksek çıkıyor.\n\nAncak küçük sınıfları (1 ve 2)\nhiç bilemediği için\nMacro-F1 (%30) gerçeği ifşa ediyor!", 
         fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round'), ha='center')

plt.tight_layout()
plt.savefig('sekil_13_micro_vs_macro.png', dpi=300)
plt.show()