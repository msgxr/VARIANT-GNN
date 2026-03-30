# VARIANT-GNN: GATv2 ve Belirsizlik Analizi Teorisi

Bu doküman, sistemin akademik temelini oluşturan **GATv2 (Graph Attention Networks v2)**, **MC-Dropout (Monte Carlo Dropout)** ve **Biyolojik Zenginleştirme** teorilerini açıklamaktadır.

## 1. GATv2: Dinamik Dikkat Mekanizması

Geleneksel GAT (Graph Attention Network) modellerinde dikkat ağırlıkları "statik" kalmaktadır. VARIANT-GNN, bu sorunu çözen **GATv2** mimarisini kullanır. Bir $i$ düğümü ile $j$ komşusu arasındaki dikkat katsayısı $\alpha_{ij}$ şu formülle hesaplanır:

$$e(h_i, h_j) = \vec{a}^T \cdot \text{LeakyReLU} \left( W \cdot [h_i \parallel h_j] \right)$$
$$\alpha_{ij} = \frac{\exp(e(h_i, h_j))}{\sum_{k \in \mathcal{N}(i)} \exp(e(h_i, h_k))}$$

GATv2'de $[h_i \parallel h_j]$ işlemi sayesinde, nükleotid dizilimindeki bir varyantın etkisi, sadece kendi değerine değil, komşu varyantların değerlerine bağlı olarak **dinamik** olarak değişir.

## 2. Belirsizlik Analizi (Uncertainty Quantification)

Kritik sağlık kararlarında "bilmiyorum" diyebilmek, tahminde bulunmak kadar önemlidir. **Monte Carlo Dropout** yöntemi ile modelin "Güven Skoru" hesaplanır:

1. **Inference Fazında Dropout:** Modelin nöronlarının bir kısmı rastgele kapatılarak $T$ sayıda ($T=15$) tahmin üretilir.
2. **Varyans Hesaplama:** Bu $T$ tahminin standart sapması ($\sigma$), modelin belirsizliğini temsil eder.
3. **Güven Skoru:** $(1 - \sigma) \times 100$ formülü ile klinisyene sunulur.

## 3. Biyolojik Zenginleştirme (BLOSUM62 & Grantham)

Sadece koordinat veya popülasyon sıklığı değil, amino asit değişiminin **biyokimyasal şiddeti** de modelin girdisidir.

- **BLOSUM62 ($B_{ij}$):** Evrimsel süreçte amino asit $i \rightarrow j$ dönüşümünü skorlar. Negatif değerler hayati fonksiyon bozukluğuna (SOTA patojenite) işaret eder.
- **Grantham Skoru ($d_{ij}$):** İki amino asit arasındaki kompozisyon, polarite ve moleküler hacim farkını ölçer:
  $$d_{ij} = \sqrt{\alpha(c_i - c_j)^2 + \beta(p_i - p_j)^2 + \gamma(v_i - v_j)^2}$$

## 4. Akış Özeti

Sistem, varyantı önce biyokimyasal olarak zenginleştirir, ardından GATv2 ile etkileşimli özellik grafında analiz eder ve son olarak MC-Dropout ile güvenilir bir "Klinik Karar Skoru" üretir. Bu hibrit yaklaşım, TEKNOFEST 2026 jürisine sunulacak olan **SOTA (State-of-the-Art)** bilimsel temeli oluşturur.
