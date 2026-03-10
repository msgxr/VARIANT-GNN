"""
src/explainability/graph_viz.py
GNN etkileşim grafını Matplotlib + NetworkX ile görselleştirir.
Preprocessor'dan gelen edge_index ve feature matrisini kullanır.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    import torch

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

try:
    import networkx as nx
    _NX_AVAILABLE = True
except ImportError:
    _NX_AVAILABLE = False
    logger.warning("networkx bulunamadı — GNN grafı görselleştirme devre dışı.")


def _build_graph(
    edge_index: torch.Tensor,
    node_features: np.ndarray,
    feature_names: List[str],
    risk_scores: Optional[np.ndarray] = None,
) -> Tuple[Optional["nx.Graph"], List[str]]:
    """NetworkX grafını edge_index üzerinden oluşturur."""
    if not _NX_AVAILABLE:
        return None, []

    n_nodes = node_features.shape[0]
    G = nx.Graph()

    # Düğümleri ekle
    for i in range(n_nodes):
        label = feature_names[i] if i < len(feature_names) else f"F{i}"
        # Özellik istatistikleri
        feat_mean = float(np.mean(node_features[:, i])) if node_features.shape[0] > 1 else float(node_features[0, i])
        G.add_node(i, label=label, mean_val=feat_mean)

    # Kenarları ekle
    ei = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.array(edge_index)
    for j in range(ei.shape[1]):
        src, dst = int(ei[0, j]), int(ei[1, j])
        if src != dst and src < n_nodes and dst < n_nodes:
            G.add_edge(src, dst)

    labels = [G.nodes[i].get("label", f"F{i}") for i in range(n_nodes)]
    return G, labels


def plot_variant_graph(
    edge_index,
    node_features: np.ndarray,
    feature_names: List[str],
    risk_scores: Optional[np.ndarray] = None,
    highlight_node: Optional[int] = None,
    top_n_nodes: int = 20,
    figsize: Tuple[int, int] = (10, 7),
) -> Optional[plt.Figure]:
    """
    Preprocesör'ün ürettiği genetik etkileşim grafını çizer.

    Parameters
    ----------
    edge_index     : PyG formatındaki kenar bilgisi (2 x E Tensor veya array)
    node_features  : (N_samples, N_features) matris — her sütun bir düğüm
    feature_names  : Özellik adları listesi
    risk_scores    : Varyant bazlı risk skorları (varsa düğüm rengi için)
    highlight_node : Vurgulanacak düğüm indeksi
    top_n_nodes    : En fazla kaç düğüm gösterilsin
    figsize        : Figür boyutu

    Returns
    -------
    Matplotlib Figure objesi veya None (networkx yoksa)
    """
    if not _NX_AVAILABLE:
        return None

    # Büyük graflarda yalnızca en yüksek degree'li düğümleri göster
    ei_arr = edge_index.cpu().numpy() if hasattr(edge_index, "cpu") else np.array(edge_index)
    n_nodes = node_features.shape[1]

    # Her düğümün bağlantı sayısını hesapla
    degree = np.zeros(n_nodes, dtype=int)
    for j in range(ei_arr.shape[1]):
        s, d = int(ei_arr[0, j]), int(ei_arr[1, j])
        if s < n_nodes:
            degree[s] += 1
        if d < n_nodes:
            degree[d] += 1

    # Top-N düğüm seçimi (highlight_node her zaman dahil)
    top_idx = set(np.argsort(degree)[::-1][:top_n_nodes].tolist())
    if highlight_node is not None and highlight_node < n_nodes:
        top_idx.add(highlight_node)
    top_idx = sorted(top_idx)

    # Yeniden indeksle
    idx_map = {old: new for new, old in enumerate(top_idx)}
    sub_features = node_features[:, top_idx] if node_features.ndim == 2 else node_features[top_idx]
    sub_names = [feature_names[i] if i < len(feature_names) else f"F{i}" for i in top_idx]

    # Kenarları filtrele
    sub_edges = []
    for j in range(ei_arr.shape[1]):
        s, d = int(ei_arr[0, j]), int(ei_arr[1, j])
        if s in idx_map and d in idx_map and s != d:
            sub_edges.append((idx_map[s], idx_map[d]))

    # NetworkX grafını oluştur
    G = nx.Graph()
    G.add_nodes_from(range(len(top_idx)))
    G.add_edges_from(sub_edges)

    # ── Görselleştirme ──────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Layout
    try:
        pos = nx.spring_layout(G, seed=42, k=1.5 / max(len(top_idx) ** 0.5, 1))
    except Exception:
        pos = nx.circular_layout(G)

    # Düğüm renkleri — risk skoruna göre
    node_colors = []
    new_hl = idx_map.get(highlight_node, -1) if highlight_node is not None else -1

    # Her düğüm için ortalama özellik değerini normalize et (renk için)
    if sub_features.ndim == 2 and sub_features.shape[0] > 0:
        col_means = np.mean(sub_features, axis=0)
    else:
        col_means = sub_features

    # Min-max normalize
    mn, mx = col_means.min(), col_means.max()
    norms = (col_means - mn) / (mx - mn + 1e-9)

    for i in range(len(top_idx)):
        if i == new_hl:
            node_colors.append('#fc8181')   # Seçili varyant — kırmızı
        else:
            # Mavi → yeşil gradient (düşük → yüksek değer)
            r = 0.2 + 0.3 * norms[i]
            g = 0.5 + 0.4 * norms[i]
            b = 0.8 - 0.3 * norms[i]
            node_colors.append((r, g, b, 0.9))

    # Kenarlar
    if G.number_of_edges() > 0:
        nx.draw_networkx_edges(
            G, pos, ax=ax,
            edge_color='#4a5568',
            alpha=0.5,
            width=0.8,
        )

    # Düğümler
    node_sizes = [
        600 if i == new_hl else 300 + int(degree[top_idx[i]] * 40)
        for i in range(len(top_idx))
    ]
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=node_sizes,
    )

    # Etiketler — uzun isimler kısalt
    short_labels = {
        i: (name[:12] + "…" if len(name) > 13 else name)
        for i, name in enumerate(sub_names)
    }
    nx.draw_networkx_labels(
        G, pos, labels=short_labels, ax=ax,
        font_size=7, font_color='#e2e8f0',
        font_weight='normal',
    )

    ax.set_title(
        f"Genetik Etkileşim Grafı — Top-{len(top_idx)} Özellik Düğümü",
        fontsize=12, fontweight='bold', color='#e2e8f0', pad=14,
    )
    ax.axis('off')

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#fc8181', label='Seçili Varyant'),
        mpatches.Patch(facecolor=(0.2, 0.7, 0.7, 0.9), label='Yüksek Etki'),
        mpatches.Patch(facecolor=(0.2, 0.5, 0.8, 0.9), label='Düşük Etki'),
    ]
    ax.legend(
        handles=legend_elements,
        loc='lower right', fontsize=8,
        facecolor='#1a2744', edgecolor='#4a5568',
        labelcolor='#94a3b8',
    )

    node_count = G.number_of_nodes()
    edge_count = G.number_of_edges()
    ax.text(
        0.01, 0.01,
        f"Düğüm: {node_count}  |  Kenar (Korelasyon Bağı): {edge_count}",
        transform=ax.transAxes,
        fontsize=7.5, color='#718096',
    )

    plt.tight_layout()
    return fig


def plot_feature_correlation_heatmap(
    node_features: np.ndarray,
    feature_names: List[str],
    figsize: Tuple[int, int] = (10, 8),
    top_n: int = 20,
) -> Optional[plt.Figure]:
    """
    Özellikler arası korelasyon ısı haritası — GNN graf oluşumunun temelini gösterir.
    """
    n = min(top_n, node_features.shape[1], len(feature_names))
    names = feature_names[:n]
    data = node_features[:, :n]

    try:
        corr = np.corrcoef(data.T)
    except Exception:
        return None

    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')

    # Isı haritası
    im = ax.imshow(corr, cmap='RdYlBu_r', aspect='auto', vmin=-1, vmax=1)

    # Eksenleri ayarla
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    short = [nm[:10] + "…" if len(nm) > 11 else nm for nm in names]
    ax.set_xticklabels(short, rotation=45, ha='right', fontsize=7, color='#94a3b8')
    ax.set_yticklabels(short, fontsize=7, color='#94a3b8')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(colors='#94a3b8', labelsize=7)
    cbar.set_label('Korelasyon', color='#94a3b8', fontsize=8)

    ax.set_title(
        f"Özellik Korelasyon Matrisi (GNN Graf Temeli, Top-{n})",
        fontsize=11, fontweight='bold', color='#e2e8f0', pad=12,
    )

    for sp in ax.spines.values():
        sp.set_edgecolor('#4a5568')

    plt.tight_layout()
    return fig
