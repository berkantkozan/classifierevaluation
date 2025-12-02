import matplotlib.pyplot as plt
import numpy as np

# 1. ADIM: VERİ HAZIRLIĞI
# Mükemmel kalibrasyon referansı (x = y)
x_perfect = np.linspace(0, 1, 100)
y_perfect = x_perfect

# Senaryoya Uygun "Kötü" Model Verisi (Aşırı Özgüvenli)
# Model yüksek olasılıklar (0.8, 0.9) tahmin ediyor ama gerçek oranlar düşük kalıyor.
prob_pred = np.array([0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0]) # Tahminler
prob_true = np.array([0.0, 0.05, 0.15, 0.2, 0.25, 0.2, 0.3, 0.4]) # Gerçekleşenler
# Dikkat: 0.8 tahminine karşılık 0.2 gerçek değer (Örnekteki durum)

# 2. ADIM: GRAFİK ÇİZİMİ
plt.figure(figsize=(8, 8))

# A. Mükemmel Kalibrasyon Çizgisi (Gri Kesik Çizgi)
plt.plot(x_perfect, y_perfect, linestyle='--', color='gray', 
         label='Mükemmel Kalibrasyon (İdeal)')

# B. Modelin Gerçek Eğrisi (Kırmızı)
plt.plot(prob_pred, prob_true, marker='o', linewidth=2, color='#e74c3c', 
         label='Model Performansı (Kalibresiz)')

# C. Örnek Durumun İşaretlenmesi (Siyah X)
# (0.8, 0.2) noktasına işaret koyuyoruz
plt.plot(0.8, 0.2, marker='X', color='black', markersize=14, label='Örnek Vaka')

# Ok ve Açıklama Kutusu Ekleme
plt.annotate('Örnek Senaryo:\nModel %80 Risk dedi,\nGerçekleşme %20 oldu.\n(Aşırı Özgüven)', 
             xy=(0.8, 0.2), xytext=(0.45, 0.4),
             arrowprops=dict(facecolor='black', shrink=0.05),
             fontsize=11, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9))

# 3. ADIM: ETİKETLER VE AYARLAR
plt.xlabel('Tahmin Edilen Olasılık (Predicted Probability)', fontsize=12)
plt.ylabel('Gerçek Pozitif Oranı (Actual Fraction)', fontsize=12)
plt.title('Şekil: Kalibrasyon Eğrisi (Reliability Diagram)', fontsize=14, pad=15)
plt.legend(loc='upper left', fontsize=11)
plt.grid(True, alpha=0.3)
plt.xlim(0, 1)
plt.ylim(0, 1)

# 4. ADIM: KAYDETME
plt.tight_layout()
plt.savefig('kalibrasyon_egrisi.png', dpi=300)
plt.show()

print("Kalibrasyon grafiği başarıyla üretildi.")