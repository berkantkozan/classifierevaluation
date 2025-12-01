import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, cohen_kappa_score, recall_score

# SENARYO:
# Pozitif Sınıf (Sağlam): 100 tane
# Negatif Sınıf (Arızalı): 10 tane
y_true = np.array([1]*100 + [0]*10)

# MODEL TAHMİNİ (Tembel/Kötü Model):
# Pozitiflerin 90'ını bildi (TP=90, FN=10)
# Negatiflerin 9'unu kaçırdı, sadece 1'ini bildi (FP=9, TN=1)
# Yani model neredeyse her şeye "1" diyor.
y_pred = np.array([1]*90 + [0]*10 + [1]*9 + [0]*1)

# Metrikleri Hesapla
acc = accuracy_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
mcc = matthews_corrcoef(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)

# Görselleştirme
metrics = ['Recall', 'F1-Score', 'Accuracy', 'Kappa', 'MCC']
values = [rec, f1, acc, kappa, mcc]
colors = ['#3498db', '#3498db', '#3498db', '#e74c3c', '#c0392b'] # MCC ve Kappa Kırmızı

plt.figure(figsize=(10, 6))
bars = plt.bar(metrics, values, color=colors, edgecolor='black', alpha=0.9)

# Değerleri çubukların üzerine yaz
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
             f'{height:.2f}', ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.title('F1, Recall ve Accuracy\'nin Topluca Yanıldığı Durum\n(Model Azınlık Sınıfını Tamamen Gözden Kaçırdığında)', fontsize=14)
plt.ylabel('Skor Değeri', fontsize=12)
plt.axhline(0, color='black', linewidth=0.8)
plt.ylim(-0.1, 1.1)

# Yorum Kutusu
plt.text(3, 0.5, "Klasik metriklerin hepsi\n%80-90 bandında 'Mükemmel'\nsonuç veriyor.\n\nOysa MCC ve Kappa (0.00)\nmodelin negatifleri bulamadığını\nve rastgele çalıştığını ifşa ediyor!", 
         fontsize=11, bbox=dict(facecolor='white', alpha=0.9, edgecolor='red', boxstyle='round'), ha='center')

plt.tight_layout()
plt.savefig('sekil_11_total_failure.png', dpi=300)
plt.show()