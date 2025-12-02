import matplotlib.pyplot as plt
import numpy as np

def draw_cd_diagram():
    # SENARYO: 5 Modelin Ortalama Sıralamaları (Düşük daha iyi)
    classifiers = ['Random Forest', 'SVM', 'XGBoost', 'Logistic Reg.', 'Naive Bayes']
    ranks = np.array([1.5, 2.1, 2.3, 4.2, 4.9])
    
    # Sıralama yapalım
    sorted_indices = np.argsort(ranks)
    classifiers = [classifiers[i] for i in sorted_indices]
    ranks = ranks[sorted_indices]
    
    # Kritik Fark (CD) Değeri (Nemenyi testinden hesaplanmış varsayalım)
    cd = 1.2 

    # Çizim Alanı
    fig, ax = plt.subplots(figsize=(10, 4))
    
    # Eksen Ayarları
    min_rank = 1
    max_rank = len(classifiers)
    
    # Ana Eksen Çizgisi
    ax.hlines(0, min_rank, max_rank, color='black', linewidth=2)
    
    # Eksen İşaretleri (Ticks)
    tick_vals = np.arange(min_rank, max_rank + 1)
    ax.plot(tick_vals, np.zeros_like(tick_vals), '|', color='black', markersize=20, markeredgewidth=2)
    for val in tick_vals:
        ax.text(val, -0.4, str(val), ha='center', va='top', fontsize=12)
        
    ax.text(min_rank, -0.9, 'En İyi Sıralama (1)', ha='center', fontweight='bold')
    ax.text(max_rank, -0.9, 'En Kötü Sıralama', ha='center', fontweight='bold')
    
    # Modelleri Yerleştirme
    for i, (name, r) in enumerate(zip(classifiers, ranks)):
        # Yazıların üst üste binmemesi için bir aşağı bir yukarı yazalım
        y_text = 1.8 if i % 2 == 0 else 1.0
        
        # Kesik çizgi ile konumu göster
        ax.plot([r, r], [0, y_text], color='gray', linestyle='--', linewidth=1)
        # Etiket Kutusu
        ax.text(r, y_text + 0.1, f"{name}\n({r:.1f})", ha='center', va='bottom', fontsize=10, fontweight='bold',
                bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.2'))
        # Nokta koy
        ax.plot(r, 0, 'o', color='black', markersize=6)

    # "Kritik Fark" (CD) Çubuğunu Gösterme (Sol üst köşe)
    cd_x_start = min_rank
    cd_x_end = min_rank + cd
    
    ax.hlines(2.5, cd_x_start, cd_x_end, color='red', linewidth=3)
    ax.text((cd_x_start + cd_x_end)/2, 2.6, f'Kritik Fark (CD) = {cd}', ha='center', va='bottom', color='red', fontweight='bold')
    ax.plot([cd_x_start, cd_x_end], [2.5, 2.5], '|', color='red', markersize=10, markeredgewidth=2)

    # Grupları (İstatistiki Olarak Farksız Olanları) Bağlama
    # Grup 1: RF, SVM, XGBoost (Farkları < 1.2)
    # 2.3 - 1.5 = 0.8 (< 1.2) -> Bunlar farksızdır.
    ax.hlines(0.3, 1.5, 2.3, linewidth=5, color='blue')
    ax.text(1.9, 0.45, 'İstatistiksel Olarak Benzer', ha='center', va='bottom', fontsize=9, color='blue', fontweight='bold')
    
    # Grup 2: Logistic Reg, Naive Bayes
    # 4.9 - 4.2 = 0.7 (< 1.2) -> Bunlar da farksızdır.
    ax.hlines(0.3, 4.2, 4.9, linewidth=5, color='blue')

    # Temizlik
    ax.axis('off')
    
    plt.title('CD (Critical Difference) Diyagramı Örneği\n(Nemenyi Testi Sonrası Görselleştirme)', fontsize=14, y=1.0)
    plt.tight_layout()
    plt.savefig('cd_diyagrami.png', dpi=300)
    plt.show()

draw_cd_diagram()