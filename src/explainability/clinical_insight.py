"""
src/explainability/clinical_insight.py
Klinik Karar Destek Asistanı — SHAP değerlerini kullanarak
varyant bazında otomatik Türkçe klinik yorum üretir.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────
# Klinik terminoloji sözlüğü
# Özellik gruplarına göre biyolojik yorumlama şablonları
# ─────────────────────────────────────────────────────────────
_FEATURE_GROUPS: Dict[str, Dict] = {
    # Amino asit ve protein etkileri
    "amino_acid": {
        "keywords": ["aa_change", "amino", "missense", "nonsense", "frameshift",
                     "polyphen", "sift", "provean", "mut_taster", "mut_assessor"],
        "label": "amino asit/protein yapısı değişimi",
        "risk_text": "Bu varyant proteinin yapısal veya fonksiyonel bütünlüğünü olumsuz etkileyen "
                     "bir amino asit değişimine yol açmaktadır. SIFT/PolyPhen gibi hesaplamalı "
                     "araçların bu mutasyon için 'damaging/deleterious' tahminleri risk artışını destekler.",
        "benign_text": "Amino asit değişimi hesaplamalı araçlar tarafından genellikle 'tolerated' "
                       "ya da 'benign' olarak değerlendirilmektedir. Protein fonksiyonu korunuyor olabilir.",
    },
    # Evrimsel korunmuşluk
    "conservation": {
        "keywords": ["phylop", "phastcons", "gerp", "conservation", "cons"],
        "label": "evrimsel korunmuşluk skoru",
        "risk_text": "Varyantın bulunduğu genomik bölge evrimsel süreçte yüksek düzeyde "
                     "korunmuştur (yüksek GERP/PhyloP skoru). Korunmuş bir bölgedeki değişimin "
                     "güçlü negatif seleksiyona tabi olması, fonksiyonel önemi işaret eder.",
        "benign_text": "Etkilenen bölge evrimsel olarak az korunmuş bir bölgede yer almaktadır. "
                       "Bu durum varyantın işlevsel açıdan daha az kritik bir konumda olduğuna işaret edebilir.",
    },
    # Popülasyon frekansı
    "population": {
        "keywords": ["af", "freq", "population", "gnomad", "exac", "allele", "maf", "1000g"],
        "label": "popülasyon allel frekansı",
        "risk_text": "Varyant populasyonda çok nadir ya da hiç gözlemlenmemiş (düşük AF) bir değişimi "
                     "temsil etmektedir. Nadir varyantlar sıklıkla hastalık ilişkili varyantlarla örtüşür.",
        "benign_text": "Varyant popülasyonda nispeten yüksek frekansta bulunmaktadır (yüksek AF). "
                       "Yaygın varyantlar genellikle hastalık yapıcı olarak değerlendirilmez.",
    },
    # Hesaplamalı risk skorları
    "computational": {
        "keywords": ["cadd", "revel", "dann", "fathmm", "vest", "metasvm", "metalr",
                     "primateai", "spliceai", "score"],
        "label": "birleşik hesaplamalı risk skoru",
        "risk_text": "Birden fazla hesaplamalı yöntem (ör. CADD, REVEL) bu varyant için yüksek "
                     "patojenite skoru öngörmektedir. Bu konsensüs modellerin yüksek skorları "
                     "güçlü bir patojenite kanıtı niteliği taşır.",
        "benign_text": "Hesaplamalı risk araçları bu varyant için düşük-orta düzey patojenite puanı vermektedir.",
    },
    # Sekans özellikleri
    "sequence": {
        "keywords": ["gc_content", "cpg", "sequence", "nucleotide", "ref", "alt",
                     "transition", "transversion", "codon"],
        "label": "sekans bağlamı ve CpG içeriği",
        "risk_text": "Sekans bağlamı (CpG adaları, GC içeriği) analizi bu bölgenin metilasyon "
                     "baskılanmasına yatkın olduğunu göstermektedir. CpG'deki C→T değişimleri "
                     "insan hastalıklarında sık görülen mutasyon mekanizmasıdır.",
        "benign_text": "Sekans bağlamı özellikler açısından ortalama değerlere yakındır.",
    },
    # Splicing
    "splicing": {
        "keywords": ["splice", "donor", "acceptor", "intron", "exon", "utr"],
        "label": "splicing bölgesi etkisi",
        "risk_text": "Varyant splicing donor/akseptör bölgesine yakın konumdadır. Splicing anomalileri "
                     "transkript kaybına veya anormal protein üretimine yol açabilir.",
        "benign_text": "Varyantın splicing bölgeleri üzerinde anlamlı bir etkisinin olmadığı öngörülmektedir.",
    },
}

_RISK_ZONES = {
    "critical": (75, 100),
    "high":     (60, 75),
    "moderate": (40, 60),
    "low":      (0, 40),
}

_ZONE_LABELS = {
    "critical": ("🔴 KRİTİK RİSK", "#fc8181"),
    "high":     ("🟠 YÜKSEK RİSK", "#f6ad55"),
    "moderate": ("🟡 ORTA RİSK",   "#faf089"),
    "low":      ("🟢 DÜŞÜK RİSK",  "#68d391"),
}


def _find_group(feature_name: str) -> Optional[str]:
    """Özellik adını analiz edip hangi biyolojik gruba ait olduğunu belirler.

    Anonymous/numeric column names (e.g. Col_0, 0, feature_12) are mapped to
    'computational' as a safe fallback so that the clinical insight engine
    always produces meaningful output.
    """
    name_lower = feature_name.lower()

    # Anonymous column detection: Col_0, 0, feature_12, etc.
    if name_lower.startswith(("col_", "feature_")) or name_lower.isdigit():
        return "computational"

    for group_key, group_info in _FEATURE_GROUPS.items():
        for kw in group_info["keywords"]:
            if kw in name_lower:
                return group_key
    return None


def _risk_zone(risk_score: float) -> str:
    for zone, (lo, hi) in _RISK_ZONES.items():
        if lo <= risk_score < hi:
            return zone
    return "critical"


def generate_clinical_insight(
    risk_score: float,
    prediction: str,
    top_features: List[Tuple[str, float]],
    probability: float = 0.5,
    variant_id: Optional[str] = None,
) -> Dict:
    """
    SHAP en önemli özelliklerini ve risk skorunu kullanarak
    otomatik klinik yorumlama metni üretir.

    Parameters
    ----------
    risk_score   : Kalibre edilmiş risk skoru (0–100)
    prediction   : 'Pathogenic' veya 'Benign'
    top_features : [(feature_name, shap_value), ...] sıralı liste
    probability  : Model olasılığı (0–1)
    variant_id   : Varyant kimliği (isteğe bağlı)

    Returns
    -------
    Dict içinde 'zone', 'zone_label', 'zone_color', 'summary',
    'key_findings', 'recommendation' alanları bulunur.
    """
    is_pathogenic = prediction.lower() == "pathogenic"
    zone = _risk_zone(risk_score)
    zone_label, zone_color = _ZONE_LABELS[zone]

    # ── ID bilgisi ─────────────────────────────────────────────
    vid_text = f"**{variant_id}** varyantı" if variant_id else "Bu varyant"

    # ── Özet cümle ─────────────────────────────────────────────
    if zone == "critical":
        summary = (
            f"{vid_text}, **{risk_score:.1f}/100** klinik risk skoru ile "
            f"**KRİTİK** patojenite sınıfında değerlendirilmektedir. "
            f"Yüksek öncelikli klinik doğrulama önerilir."
        )
    elif zone == "high":
        summary = (
            f"{vid_text}, **{risk_score:.1f}/100** klinik risk skoru ile "
            f"**YÜKSEK RİSK** grubuna girmektedir. "
            f"Fonksiyonel doğrulama testleri düşünülmelidir."
        )
    elif zone == "moderate":
        summary = (
            f"{vid_text}, **{risk_score:.1f}/100** risk skoru ile "
            f"**ORTA RİSK** kategorisindedir. Ek kanıt toplanması tavsiye edilir."
        )
    else:
        summary = (
            f"{vid_text}, **{risk_score:.1f}/100** risk skoru ile "
            f"**DÜŞÜK RİSK** kategorisindedir. "
            f"Büyük olasılıkla benign bir varyantı temsil etmektedir."
        )

    # ── Kilit bulgular ──────────────────────────────────────────
    key_findings: List[Dict] = []
    seen_groups = set()

    for feat_name, shap_val in top_features[:8]:
        group = _find_group(feat_name)
        if group and group not in seen_groups:
            seen_groups.add(group)
            g_info = _FEATURE_GROUPS[group]
            insight_text = g_info["risk_text"] if is_pathogenic else g_info["benign_text"]
            direction = "artırdı" if shap_val > 0 else "azalttı"
            key_findings.append({
                "feature":   feat_name,
                "group":     g_info["label"],
                "shap":      shap_val,
                "direction": direction,
                "insight":   insight_text,
            })
        if len(key_findings) >= 3:
            break

    # En az 1 bulgu garantisi
    if not key_findings and top_features:
        feat_name, shap_val = top_features[0]
        key_findings.append({
            "feature":   feat_name,
            "group":     "hesaplamalı özellik",
            "shap":      shap_val,
            "direction": "artırdı" if shap_val > 0 else "azalttı",
            "insight":   (
                "Bu özellik, ensemble modelin temel belirleyicilerinden biridir. "
                "Değerinin tipik patojenik varyantlarla uyumu risk tahminine yansımaktadır."
            ),
        })

    # ── Öneri ──────────────────────────────────────────────────
    if zone in ("critical", "high"):
        recommendation = (
            "⚡ **Klinik Öneri:** Segregasyon analizi ve fonksiyonel biyokimyasal testler ile "
            "varyantın hastalık oluşturma potansiyeli doğrulanmalıdır. "
            "Mevcut ACMG/AMP klinik sınıflandırma kriterleri ile değerlendirilmesi önerilir."
        )
    elif zone == "moderate":
        recommendation = (
            "🔬 **Klinik Öneri:** Varyant önemi belirsizdir (VUS – Variant of Uncertain Significance). "
            "Aile tarihçesi, popülasyon veritabanları (gnomAD) ve fonksiyonel kanıtlarla birlikte "
            "kapsamlı değerlendirme yapılmalıdır."
        )
    else:
        recommendation = (
            "✅ **Klinik Öneri:** Varyant büyük olasılıkla benign ya da polimorfik niteliktedir. "
            "Klinik tablonun açıklanması için ek genomik bölgeler araştırılabilir."
        )

    return {
        "zone":           zone,
        "zone_label":     zone_label,
        "zone_color":     zone_color,
        "summary":        summary,
        "key_findings":   key_findings,
        "recommendation": recommendation,
        "probability":    probability,
        "risk_score":     risk_score,
    }
