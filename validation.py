import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- 1. GÖRSEL: TRAIN-TEST SPLIT (ŞEKİL 7) ---
def save_train_test_figure():
    fig, ax = plt.subplots(figsize=(10, 3))
    
    # Dikdörtgenleri Çiz
    # Eğitim Seti (%70) - Yeşil
    ax.add_patch(patches.Rectangle((0, 0.3), 0.7, 0.4, edgecolor='black', facecolor='#2ecc71'))
    ax.text(0.35, 0.5, 'Eğitim Seti (%70)', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Test Seti (%30) - Mavi
    ax.add_patch(patches.Rectangle((0.7, 0.3), 0.3, 0.4, edgecolor='black', facecolor='#3498db'))
    ax.text(0.85, 0.5, 'Test Seti (%30)', ha='center', va='center', color='white', fontweight='bold', fontsize=12)
    
    # Başlık ve Ayarlar
    ax.set_title('Eğitim/Test Ayrımı (Train-Test Split)', fontsize=14, pad=15)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sekil_7_train_test.png', dpi=300)
    plt.close() # Hafızayı temizle
    print("Şekil 7 (Train-Test) başarıyla kaydedildi.")

# --- 2. GÖRSEL: K-FOLD CROSS VALIDATION (ŞEKİL 8) ---
def save_k_fold_figure():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = ['#2ecc71'] * 5 # Hepsi yeşil (Eğitim)
    
    # 5 İterasyonu Çiz
    for i in range(5):
        y_pos = 5 - (i + 1)
        current_colors = colors.copy()
        current_colors[i] = '#3498db' # O anki test seti mavi olsun
        
        start = 0
        # Satır Etiketi (Sol tarafa)
        ax.text(-0.12, y_pos + 0.3, f'İterasyon {i+1}', ha='right', va='center', fontsize=11, fontweight='bold')
        
        for j, color in enumerate(current_colors):
            label = "Test" if i == j else "Eğitim"
            # Dikdörtgen
            ax.add_patch(patches.Rectangle((start, y_pos), 0.2, 0.6, edgecolor='black', facecolor=color))
            # Metin
            if i == j: # Sadece test kutusuna yazı yazalım ki kalabalık olmasın
                ax.text(start + 0.1, y_pos + 0.3, label, ha='center', va='center', color='white', fontsize=10, fontweight='bold')
            start += 0.2

    # Başlık ve Oklar
    ax.set_title('5-Katlı Çapraz Doğrulama (5-Fold CV)', fontsize=14, pad=20)
    
    # Ortalama Alma Oku
    ax.annotate('Ortalama Başarı Skoru Hesaplanır', xy=(0.5, -0.2), xytext=(0.5, -0.8),
                xycoords='data', textcoords='data',
                arrowprops=dict(facecolor='black', shrink=0.05),
                ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1, 6) # Ok için alt sınırı genişlettim
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('sekil_8_k_fold.png', dpi=300)
    plt.close()
    print("Şekil 8 (K-Fold) başarıyla kaydedildi.")

# Fonksiyonları çalıştır
save_train_test_figure()
save_k_fold_figure()