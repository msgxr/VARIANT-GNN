"""
src/features/multimodal_encoder.py
Multimodal sequence encoder for variant context.

TEKNOFEST 2026 specification requires the ±5 nucleotide and ±5 amino-acid
context around the variant site.  This module:
  1. Tokenises raw nucleotide / amino-acid strings into integer indices.
  2. Passes each sequence through a learnable Embedding layer.
  3. Extracts local motif features via a small 1D-CNN block.
  4. The resulting compact vectors are concatenated to the numeric node
     features when building the GNN input (see VariantSAGEGNN).

Usage
-----
    from src.features.multimodal_encoder import SequenceEncoder, tokenize_nucleotides, tokenize_amino_acids

    # Tokenise raw strings from the DataFrame
    nuc_ids = tokenize_nucleotides(df["nuc_context"].tolist())   # [N, 11]
    aa_ids  = tokenize_amino_acids(df["aa_context"].tolist())    # [N, 11]

    encoder = SequenceEncoder()
    seq_features = encoder(
        torch.tensor(nuc_ids),
        torch.tensor(aa_ids),
    )   # [N, encoder.output_dim]
"""
from __future__ import annotations

import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary tables
# ---------------------------------------------------------------------------

# Standard IUPAC nucleotide codes → integer index
# 0 = PAD, 5 = ambiguous / unknown
NUC_VOCAB: dict[str, int] = {
    "<PAD>": 0,
    "A":     1,
    "C":     2,
    "G":     3,
    "T":     4,
    "U":     4,   # RNA uracil mapped to T
    "N":     5, "R": 5, "Y": 5, "S": 5, "W": 5,
    "K":     5, "M": 5, "B": 5, "D": 5, "H": 5, "V": 5,
}
NUC_VOCAB_SIZE = 6   # 0..5

# Standard single-letter amino-acid codes → integer index
# 0 = PAD, 21 = unknown / non-standard
AA_VOCAB: dict[str, int] = {
    "<PAD>": 0,
    "A": 1,  "R": 2,  "N": 3,  "D": 4,  "C": 5,
    "Q": 6,  "E": 7,  "G": 8,  "H": 9,  "I": 10,
    "L": 11, "K": 12, "M": 13, "F": 14, "P": 15,
    "S": 16, "T": 17, "W": 18, "Y": 19, "V": 20,
    "X": 21, "*": 21, "-": 21, "?": 21, "U": 21, "B": 21, "Z": 21,
}
AA_VOCAB_SIZE = 22   # 0..21

# Default sequence lengths: ±5 + ref = 11
NUC_SEQ_LEN: int = 11
AA_SEQ_LEN:  int = 11


# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

def tokenize_nucleotides(
    sequences: List[str],
    seq_len:   int = NUC_SEQ_LEN,
) -> np.ndarray:
    """
    Convert a list of nucleotide strings to a padded integer token matrix.

    Parameters
    ----------
    sequences : List of nucleotide context strings (e.g. ``"ACGTTACGT"``).
    seq_len   : Output length; sequences longer than this are truncated,
                shorter ones are zero-padded.

    Returns
    -------
    np.ndarray of shape [N, seq_len] with dtype int64.
    """
    out = np.zeros((len(sequences), seq_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = str(seq).upper()[:seq_len]
        for j, char in enumerate(seq):
            out[i, j] = NUC_VOCAB.get(char, 5)   # 5 = unknown
    return out


def tokenize_amino_acids(
    sequences: List[str],
    seq_len:   int = AA_SEQ_LEN,
) -> np.ndarray:
    """
    Convert a list of amino-acid strings to a padded integer token matrix.

    Parameters
    ----------
    sequences : List of AA context strings (single-letter codes).
    seq_len   : Output length.

    Returns
    -------
    np.ndarray of shape [N, seq_len] with dtype int64.
    """
    out = np.zeros((len(sequences), seq_len), dtype=np.int64)
    for i, seq in enumerate(sequences):
        seq = str(seq).upper()[:seq_len]
        for j, char in enumerate(seq):
            out[i, j] = AA_VOCAB.get(char, 21)   # 21 = unknown
    return out


# ---------------------------------------------------------------------------
# Sequence encoder module
# ---------------------------------------------------------------------------

class SequenceEncoder(nn.Module):
    """
    Dual-branch sequence encoder mapping nucleotide and amino-acid context
    strings to a fixed-dimension feature vector.

    Architecture per branch:
      ┌─ Embedding(vocab_size, embedding_dim, padding_idx=0)
      │
      ├─ Conv1d(embedding_dim, cnn_channels, kernel=3, pad=1) → ReLU
      │
      ├─ Conv1d(cnn_channels, cnn_channels, kernel=3, pad=1) → ReLU
      │
      └─ AdaptiveAvgPool1d(1)  →  [N, cnn_channels]

    Outputs of both branches are concatenated: [N, cnn_channels * 2].

    Parameters
    ----------
    nuc_vocab_size : Nucleotide vocabulary size (default 6).
    aa_vocab_size  : Amino-acid vocabulary size (default 22).
    embedding_dim  : Embedding dimension for both vocabs.
    cnn_channels   : Channels in the 1D-CNN layers.
    """

    def __init__(
        self,
        nuc_vocab_size: int = NUC_VOCAB_SIZE,
        aa_vocab_size:  int = AA_VOCAB_SIZE,
        embedding_dim:  int = 8,
        cnn_channels:   int = 16,
    ) -> None:
        super().__init__()

        # ── Nucleotide branch ─────────────────────────────────────────
        self.nuc_embed = nn.Embedding(
            nuc_vocab_size, embedding_dim, padding_idx=0
        )
        self.nuc_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        # ── Amino-acid branch ─────────────────────────────────────────
        self.aa_embed = nn.Embedding(
            aa_vocab_size, embedding_dim, padding_idx=0
        )
        self.aa_cnn = nn.Sequential(
            nn.Conv1d(embedding_dim, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

        self.output_dim: int = cnn_channels * 2

        logger.info(
            "SequenceEncoder: embedding_dim=%d, cnn_channels=%d, output_dim=%d",
            embedding_dim, cnn_channels, self.output_dim,
        )

    def forward(
        self,
        nuc_ids: torch.Tensor,   # [N, nuc_seq_len] — int64
        aa_ids:  torch.Tensor,   # [N, aa_seq_len]  — int64
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        nuc_ids : [N, nuc_seq_len] nucleotide token indices.
        aa_ids  : [N, aa_seq_len] amino-acid token indices.

        Returns
        -------
        torch.Tensor of shape [N, output_dim] (float32).
        """
        # Nucleotide
        nuc_emb  = self.nuc_embed(nuc_ids).permute(0, 2, 1)   # [N, emb, seq]
        nuc_feat = self.nuc_cnn(nuc_emb).squeeze(-1)            # [N, cnn_ch]

        # Amino acid
        aa_emb   = self.aa_embed(aa_ids).permute(0, 2, 1)
        aa_feat  = self.aa_cnn(aa_emb).squeeze(-1)

        return torch.cat([nuc_feat, aa_feat], dim=1)            # [N, output_dim]
