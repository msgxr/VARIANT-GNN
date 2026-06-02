"""
src/scientific/pubmed_rag.py
PubMed Canlı Makale Çekim Altyapısı (RAG Modu)

Varyant ID veya gen adı için NCBI PubMed'den güncel makaleleri çeker,
özet cümleleri çıkarır ve PDF rapora entegre edilebilir yapıda döndürür.

RAG (Retrieval-Augmented Generation) prensibi:
  Sorgu → PubMed E-utilities → Özet → Anahtar cümle çıkarımı → PDF

NCBI Politikası:
  - API anahtarsız: max 3 istek/saniye
  - API anahtarlı (NCBI_API_KEY env): max 10 istek/saniye
  - Akademik/araştırma amaçlı kullanım: ücretsiz

Kullanım:
    rag = PubMedRAG()
    articles = rag.fetch(gene="BRCA1", n_results=3)
    snippet  = rag.fetch_for_variant(variant_id="BRCA1-c.5266dup", gene="BRCA1")
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_NCBI_KEY = os.environ.get("NCBI_API_KEY", "")
_DELAY = 0.35 if _NCBI_KEY else 1.0  # Rate limit


def _ncbi_get(url: str, timeout: int = 8) -> Optional[dict]:
    """NCBI API'ye GET isteği atar; JSON döner veya None."""
    if _NCBI_KEY:
        url += f"&api_key={_NCBI_KEY}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # nosec B310
            return json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError) as exc:
        logger.warning("NCBI GET hatası (%s): %s", url[:80], exc)
        return None


def _extract_key_sentence(abstract: str, keywords: List[str]) -> str:
    """Özetten anahtar kelime içeren en alakalı cümleyi çıkarır."""
    if not abstract:
        return ""
    sentences = [s.strip() for s in abstract.replace("\n", " ").split(".") if s.strip()]
    kw_lower = [k.lower() for k in keywords]

    scored = []
    for sent in sentences:
        score = sum(1 for kw in kw_lower if kw in sent.lower())
        scored.append((score, sent))

    scored.sort(key=lambda t: t[0], reverse=True)
    if scored and scored[0][0] > 0:
        return scored[0][1][:300] + ("…" if len(scored[0][1]) > 300 else "")
    return sentences[0][:300] if sentences else ""


class PubMedRAG:
    """
    PubMed makale çekici ve RAG özet üretici.

    Parameters
    ----------
    cache_ttl : Önbellekte tutma süresi (saniye). 0 = önbelleksiz.
    """

    def __init__(self, cache_ttl: int = 3600) -> None:
        self._cache: Dict[str, tuple] = {}  # query → (timestamp, result)
        self._cache_ttl = cache_ttl

    def _cached(self, key: str):
        if key in self._cache:
            ts, val = self._cache[key]
            if self._cache_ttl == 0 or (time.time() - ts) < self._cache_ttl:
                return val
        return None

    def _store(self, key: str, val) -> None:
        self._cache[key] = (time.time(), val)

    # ── Arama ────────────────────────────────────────────────────────────────
    def search_pmids(self, query: str, n: int = 5) -> List[str]:
        """
        PubMed'de arama yap, PMID listesi döndür.

        Parameters
        ----------
        query : Arama terimi (örn. "BRCA1 pathogenic variant")
        n     : Kaç sonuç istendiği
        """
        cache_key = f"pmids::{query}::{n}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        url = (
            f"{_BASE_URL}/esearch.fcgi"
            f"?db=pubmed&term={urllib.parse.quote(query)}"
            f"&retmax={n}&retmode=json&sort=relevance"
        )
        data = _ncbi_get(url)
        pmids = []
        if data:
            pmids = data.get("esearchresult", {}).get("idlist", [])

        self._store(cache_key, pmids)
        time.sleep(_DELAY)
        return pmids

    def fetch_summaries(self, pmids: List[str]) -> List[dict]:
        """
        PMID listesi için özet bilgileri çek (başlık, yıl, dergi, URL).
        """
        if not pmids:
            return []

        cache_key = f"summ::{','.join(pmids)}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        url = f"{_BASE_URL}/esummary.fcgi?db=pubmed&id={','.join(pmids)}&retmode=json"
        data = _ncbi_get(url)
        results = []

        if data:
            result_map = data.get("result", {})
            for pmid in pmids:
                rec = result_map.get(pmid, {})
                if not rec:
                    continue
                authors = rec.get("authors", [])
                first_author = authors[0].get("name", "") if authors else "?"
                results.append(
                    {
                        "pmid": pmid,
                        "title": rec.get("title", "").rstrip("."),
                        "year": rec.get("pubdate", "")[:4],
                        "journal": rec.get("source", ""),
                        "authors": f"{first_author} et al." if len(authors) > 1 else first_author,
                        "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                        "doi": next(
                            (art.get("value", "") for art in rec.get("articleids", []) if art.get("idtype") == "doi"),
                            "",
                        ),
                    }
                )

        time.sleep(_DELAY)
        self._store(cache_key, results)
        return results

    def fetch_abstracts(self, pmids: List[str]) -> Dict[str, str]:
        """PMID → abstract metin sözlüğü."""
        if not pmids:
            return {}

        cache_key = f"abs::{','.join(pmids)}"
        cached = self._cached(cache_key)
        if cached is not None:
            return cached

        url = f"{_BASE_URL}/efetch.fcgi?db=pubmed&id={','.join(pmids)}&rettype=abstract&retmode=text"
        try:
            with urllib.request.urlopen(url, timeout=10) as r:  # nosec B310
                raw = r.read().decode("utf-8")
        except Exception as exc:
            logger.warning("Abstract çekim hatası: %s", exc)
            return {}

        # Basit parse: her PMID için ilk abstract bloğu
        abstracts: Dict[str, str] = {}
        blocks = raw.split("\n\n")
        for pmid in pmids:
            for block in blocks:
                if f"PMID: {pmid}" in block or pmid in block:
                    abstracts[pmid] = block[:800]
                    break

        time.sleep(_DELAY)
        self._store(cache_key, abstracts)
        return abstracts

    # ── Yüksek Seviye API ────────────────────────────────────────────────────
    def fetch(
        self,
        gene: str,
        variant: Optional[str] = None,
        n_results: int = 3,
        keywords: Optional[List[str]] = None,
    ) -> List[dict]:
        """
        Bir gen (ve opsiyonel varyant) için alakalı makaleleri çek.

        Returns
        -------
        list of {pmid, title, year, journal, authors, url, abstract_snippet}
        """
        query_parts = [gene, "pathogenicity", "variant"]
        if variant:
            query_parts.append(variant)
        if keywords:
            query_parts.extend(keywords[:2])
        query = " ".join(query_parts)

        pmids = self.search_pmids(query, n=n_results)
        if not pmids:
            logger.info("PubMed sonuç bulunamadı: %s", query)
            return []

        summaries = self.fetch_summaries(pmids)
        abstracts = self.fetch_abstracts(pmids)

        kws = [gene] + (keywords or ["pathogenic", "variant", "clinical"])
        results = []
        for s in summaries:
            pmid = s["pmid"]
            abs_text = abstracts.get(pmid, "")
            snippet = _extract_key_sentence(abs_text, kws)
            results.append({**s, "abstract_snippet": snippet})

        logger.info(
            "PubMed: '%s' → %d makale getirildi (PMID: %s)",
            query,
            len(results),
            [r["pmid"] for r in results],
        )
        return results

    def fetch_for_variant(
        self,
        variant_id: str,
        gene: Optional[str] = None,
        n_results: int = 3,
    ) -> List[dict]:
        """
        Varyant ID ve gen adına göre PubMed makale çek.
        PDF rapora doğrudan beslenebilir format döner.
        """
        gene_name = gene or variant_id.split("-")[0].split("_")[0]
        return self.fetch(
            gene=gene_name,
            variant=variant_id,
            n_results=n_results,
            keywords=["clinical significance", "pathogenic"],
        )

    def format_for_pdf(self, articles: List[dict]) -> str:
        """
        Makale listesini PDF'e eklenecek düz metin formatına çevirir.
        """
        if not articles:
            return "PubMed referansı bulunamadı."
        lines = []
        for i, a in enumerate(articles, 1):
            lines.append(
                f"[{i}] {a.get('authors', '?')} ({a.get('year', '?')}). "
                f"{a.get('title', '?')}. {a.get('journal', '?')}. "
                f"PMID:{a.get('pmid', '?')}"
            )
            if a.get("abstract_snippet"):
                lines.append(f"    → {a['abstract_snippet']}")
        return "\n".join(lines)
