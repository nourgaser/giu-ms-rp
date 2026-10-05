"""
models.py — the two fusion mechanisms under comparison, and nothing else.

The whole study rests on one methodological claim: that the ONLY difference
between the baseline and the proposed model is how the modalities are combined.
If anything else differs — encoder tuning, parameter count by an order of
magnitude, a different classifier head, different regularisation — then a
difference in F1 is not attributable to fusion and the result says nothing.

So both models here:

  * consume the SAME cached frozen features (see features.py),
  * project each modality to the same `d_model`,
  * end in an identical classifier head,
  * are trained by the same loop with the same hyperparameters.

They differ in the middle, and only there:

    LateFusion   pool each modality independently -> concatenate -> classify
                 This is what the Ar-MuSA and EGY-MER baselines do. The
                 modalities never see each other before the classifier.

    TriCAN       text attends to audio and to visual before pooling
                 Text is the Query; audio and visual supply Key/Value. So a
                 text token can be modified by the specific audio segments or
                 image regions that bear on it, including ones that contradict
                 it. Late fusion structurally cannot do this: by the time the
                 vectors meet, each has already been reduced to a single
                 summary and the alignment information is gone.

Parameter counts are deliberately reported by `describe()` and asserted to be
within the same order of magnitude in the tests. Tri-CAN is necessarily larger
(attention has projections that concatenation does not), and pretending
otherwise would be dishonest; what matters is that the gap is small enough that
"it just had more parameters" is not a sufficient explanation, and that the
capacity-matched control below exists.

    LateFusionWide   the baseline widened to match Tri-CAN's parameter count.
                     If Tri-CAN beats plain LateFusion but not this, the gain
                     was capacity, not cross-modal attention. This control is
                     the difference between a result and a press release.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskedMeanPool(nn.Module):
    """Mean over the sequence axis, ignoring padded positions."""

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, L, D)   mask: (B, L) with True = real token
        if mask is None:
            return x.mean(dim=1)
        m = mask.unsqueeze(-1).to(x.dtype)
        return (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class _Head(nn.Module):
    """The classifier head. Identical across every model in this file."""

    def __init__(self, d_in: int, n_labels: int, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_in),
            nn.Dropout(dropout),
            nn.Linear(d_in, d_in // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_in // 2, n_labels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _Projections(nn.Module):
    """Per-modality linear projection to a shared width. Common to both models."""

    def __init__(self, dims: dict[str, int], d_model: int) -> None:
        super().__init__()
        self.proj = nn.ModuleDict(
            {m: nn.Sequential(nn.Linear(d, d_model), nn.LayerNorm(d_model))
             for m, d in dims.items()}
        )
        self.modalities = tuple(sorted(dims))

    def forward(self, feats: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {m: self.proj[m](feats[m]) for m in self.modalities}


class LateFusion(nn.Module):
    """Baseline: independent encoding, concatenation at the classifier.

    This reproduces the fusion strategy of the published Ar-MuSA and EGY-MER
    baselines. It is the thing to beat, and it is implemented here rather than
    quoted from a paper so that it runs on identical features.
    """

    name = "late-fusion"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = _Projections(dims, d_model)
        self.pool = MaskedMeanPool()
        self.head = _Head(d_model * len(dims), n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks or {}
        p = self.proj(feats)
        pooled = [self.pool(p[m], masks.get(m)) for m in self.proj.modalities]
        return self.head(torch.cat(pooled, dim=-1))


class LateFusionWide(LateFusion):
    """Capacity-matched control — late fusion with a wider projection.

    Exists solely to answer "isn't Tri-CAN just bigger?". `d_model` is chosen
    by `match_capacity()` below so the parameter counts land within a few
    percent of each other.
    """

    name = "late-fusion-wide"


class CrossModalBlock(nn.Module):
    """One direction of cross-modal attention: Query from text, K/V from other.

    Standard post-norm transformer cross-attention. Nothing novel — and the
    write-up says so. The contribution of this project is not the block; it is
    the question of whether the block still helps under a different task regime
    and on samples where the modalities disagree.
    """

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.1) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model * 2, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                kv_mask: torch.Tensor | None = None,
                need_weights: bool = False):
        # nn.MultiheadAttention wants True = IGNORE, our masks are True = KEEP.
        pad = (~kv_mask) if kv_mask is not None else None
        a, w = self.attn(q, kv, kv, key_padding_mask=pad,
                         need_weights=need_weights, average_attn_weights=True)
        x = self.norm1(q + self.drop(a))
        x = self.norm2(x + self.drop(self.ff(x)))
        return (x, w) if need_weights else (x, None)


class TriCAN(nn.Module):
    """Tri-modal cross-attention: text queries audio and visual, then fuses.

    Text is the anchor because it is the modality with the strongest encoder
    for Arabic (AraBERT) and because the motivating phenomenon — words
    contradicted by delivery — is naturally expressed as "what does the audio
    or the face say about THIS span of text".

    Note on the visual stream for Ar-MuSA: the corpus provides a single frame
    per clip, so the visual token sequence is Swin's SPATIAL patch grid, not a
    temporal sequence. Text therefore attends over image regions. That is a
    coherent operation, but it is a weaker signal than attending over a video,
    and the write-up must not describe it as temporal.

    `return_attention=True` yields the attention maps, which is how the
    qualitative incongruence analysis is produced.
    """

    name = "tri-can"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 anchor: str = "text") -> None:
        super().__init__()
        if anchor not in dims:
            raise ValueError(f"anchor {anchor!r} not among modalities {sorted(dims)}")
        self.anchor = anchor
        self.others = tuple(m for m in sorted(dims) if m != anchor)

        self.proj = _Projections(dims, d_model)
        self.blocks = nn.ModuleDict(
            {m: CrossModalBlock(d_model, n_heads, dropout=0.1) for m in self.others}
        )
        self.pool = MaskedMeanPool()
        # anchor + one attended stream per other modality
        self.head = _Head(d_model * (1 + len(self.others)), n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None,
                return_attention: bool = False):
        masks = masks or {}
        p = self.proj(feats)
        q = p[self.anchor]
        q_mask = masks.get(self.anchor)

        pooled = [self.pool(q, q_mask)]
        attn_maps: dict[str, torch.Tensor] = {}

        for m in self.others:
            out, w = self.blocks[m](q, p[m], masks.get(m), need_weights=return_attention)
            pooled.append(self.pool(out, q_mask))
            if return_attention and w is not None:
                attn_maps[m] = w.detach()

        logits = self.head(torch.cat(pooled, dim=-1))
        return (logits, attn_maps) if return_attention else logits


# -- construction helpers -------------------------------------------------

class HybridFusion(nn.Module):
    """Late fusion AND cross-attention, rather than one instead of the other.

    This exists because of a design flaw in Tri-CAN that the EGY-MER results
    exposed. Tri-CAN feeds its classifier

        [ pool(anchor), pool(anchor's view of A), pool(anchor's view of B) ]

    where every term is pooled over ANCHOR positions. All three therefore live
    in the anchor's token space, and the raw A and B representations never
    reach the classifier. Tri-CAN is not "late fusion plus attention" — it
    *replaces* the other modalities with the anchor's interrogation of them,
    and so discards information plain late fusion keeps.

    That predicts the observed pattern precisely. When the modalities agree,
    the anchor's view of the others is serviceable and attention adds context
    (+0.036 macro-F1 on congruent EGY-MER samples). When they disagree, the
    anchor is exactly the channel that is wrong, and the classifier has no
    unmediated access to the others (-0.047 on incongruent samples).

    The fix keeps both:

        [ pool(anchor), pool(A), pool(B),
          pool(anchor's view of A), pool(anchor's view of B) ]

    which is a strict superset of late fusion's input. Attention can now only
    add evidence, never replace it — so it should at worst tie late fusion,
    and should beat it wherever the cross-modal context genuinely helps.

    `gate=True` additionally learns a scalar per attended stream, so the model
    can suppress the attended view on samples where it is unhelpful instead of
    being forced to consume it.
    """

    name = "hybrid"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 anchor: str = "text", gate: bool = False) -> None:
        super().__init__()
        if anchor not in dims:
            raise ValueError(f"anchor {anchor!r} not among modalities {sorted(dims)}")
        self.anchor = anchor
        self.mods = tuple(sorted(dims))
        self.others = tuple(m for m in self.mods if m != anchor)

        self.proj = _Projections(dims, d_model)
        self.blocks = nn.ModuleDict(
            {m: CrossModalBlock(d_model, n_heads, dropout=0.1) for m in self.others}
        )
        self.pool = MaskedMeanPool()
        self.use_gate = gate
        if gate:
            self.gates = nn.Sequential(
                nn.Linear(d_model * len(self.mods), d_model), nn.GELU(),
                nn.Linear(d_model, len(self.others)), nn.Sigmoid(),
            )
        n_parts = len(self.mods) + len(self.others)
        self.head = _Head(d_model * n_parts, n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None,
                return_attention: bool = False):
        masks = masks or {}
        p = self.proj(feats)
        q, q_mask = p[self.anchor], masks.get(self.anchor)

        # 1. every modality's own pooled state — what late fusion sees
        own = [self.pool(p[m], masks.get(m)) for m in self.mods]

        # 2. the anchor's attended view of each other modality
        attended, attn_maps = [], {}
        for m in self.others:
            out, w = self.blocks[m](q, p[m], masks.get(m),
                                    need_weights=return_attention)
            attended.append(self.pool(out, q_mask))
            if return_attention and w is not None:
                attn_maps[m] = w.detach()

        if self.use_gate:
            g = self.gates(torch.cat(own, dim=-1))
            attended = [a * g[:, i:i + 1] for i, a in enumerate(attended)]

        logits = self.head(torch.cat(own + attended, dim=-1))
        return (logits, attn_maps) if return_attention else logits


class GatedFusion(nn.Module):
    """Gated late fusion — the fusion strategy the EGY-MER paper actually uses.

    Our plain `LateFusion` concatenates pooled modality vectors. The published
    EGY-MER baseline (weighted F1 0.519) instead learns a per-modality *gate*
    that scales each stream's contribution before combining, so an
    uninformative modality can be attenuated for a given sample. Comparing
    against plain concatenation and calling it "the published baseline" would
    be inaccurate, so it is implemented here.

    The gate is computed from all modalities jointly — a modality's weight
    should depend on what the others are saying, which is the entire point.
    """

    name = "gated-fusion"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = _Projections(dims, d_model)
        self.pool = MaskedMeanPool()
        n = len(dims)
        self.gate = nn.Sequential(
            nn.Linear(d_model * n, d_model), nn.GELU(),
            nn.Linear(d_model, n), nn.Sigmoid(),
        )
        self.head = _Head(d_model * n, n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks or {}
        p = self.proj(feats)
        pooled = [self.pool(p[m], masks.get(m)) for m in self.proj.modalities]
        cat = torch.cat(pooled, dim=-1)
        g = self.gate(cat)                                  # (B, n_modalities)
        gated = [pooled[i] * g[:, i:i + 1] for i in range(len(pooled))]
        return self.head(torch.cat(gated, dim=-1))


class AllPairsCrossAttention(nn.Module):
    """Every modality attends to every other — the MulT arrangement.

    Tri-CAN picks one anchor and throws away the other directions. MulT
    (Tsai et al. 2019), the paper the whole approach descends from, keeps all
    of them: with three modalities there are six directed pairs, each modality
    receiving from the two others. The per-modality results are concatenated.

    This removes the anchor choice entirely, which matters because that choice
    turned out to be wrong for this corpus. It costs more parameters, so the
    comparison against it must be read alongside the capacity-matched control.
    """

    name = "all-pairs"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3) -> None:
        super().__init__()
        self.proj = _Projections(dims, d_model)
        self.mods = self.proj.modalities
        self.blocks = nn.ModuleDict({
            f"{q}<-{kv}": CrossModalBlock(d_model, n_heads, dropout=0.1)
            for q in self.mods for kv in self.mods if q != kv
        })
        self.pool = MaskedMeanPool()
        # each modality contributes (its own pooled state + one per other)
        self.head = _Head(d_model * len(self.mods) * len(self.mods), n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks or {}
        p = self.proj(feats)
        parts = []
        for q in self.mods:
            parts.append(self.pool(p[q], masks.get(q)))
            for kv in self.mods:
                if kv == q:
                    continue
                out, _ = self.blocks[f"{q}<-{kv}"](p[q], p[kv], masks.get(kv))
                parts.append(self.pool(out, masks.get(q)))
        return self.head(torch.cat(parts, dim=-1))


class EarlyFusionTransformer(nn.Module):
    """Concatenate all modality tokens into one sequence and self-attend.

    The third sensible arrangement, and the one that makes no assumption at all
    about which modality should lead: every token attends to every other token
    regardless of origin. A learned modality embedding is added so the model can
    still tell the streams apart.

    Included because it is the natural "no anchor, no direction" control. If it
    matches or beats the directed variants, then the directional structure that
    Tri-CAN and MulT impose is not earning its complexity on this corpus.
    """

    name = "early-fusion"

    def __init__(self, dims: dict[str, int], n_labels: int,
                 d_model: int = 256, n_heads: int = 4, dropout: float = 0.3,
                 n_layers: int = 2) -> None:
        super().__init__()
        self.proj = _Projections(dims, d_model)
        self.mods = self.proj.modalities
        self.mod_emb = nn.Parameter(torch.randn(len(self.mods), d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model, n_heads, dim_feedforward=d_model * 2, dropout=0.1,
            batch_first=True, norm_first=False, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.pool = MaskedMeanPool()
        self.head = _Head(d_model, n_labels, dropout)

    def forward(self, feats: dict[str, torch.Tensor],
                masks: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        masks = masks or {}
        p = self.proj(feats)
        seqs, msks = [], []
        for i, m in enumerate(self.mods):
            seqs.append(p[m] + self.mod_emb[i])
            msks.append(masks.get(m, torch.ones(p[m].shape[:2], dtype=torch.bool,
                                                device=p[m].device)))
        x = torch.cat(seqs, dim=1)
        mask = torch.cat(msks, dim=1)
        # nn.TransformerEncoder wants True = IGNORE
        h = self.enc(x, src_key_padding_mask=~mask)
        return self.head(self.pool(h, mask))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def match_capacity(dims: dict[str, int], n_labels: int, target: int,
                   dropout: float = 0.3) -> LateFusionWide:
    """Widen LateFusion until its parameter count is closest to `target`.

    Searched rather than solved: the head's width depends on d_model in two
    places, so the relationship is quadratic and an exact solution would need
    rounding anyway. A scan over plausible widths is clearer to read and is
    run once per experiment.
    """
    best, best_gap = None, float("inf")
    for d in range(64, 1025, 8):
        cand = LateFusionWide(dims, n_labels, d_model=d, dropout=dropout)
        gap = abs(count_params(cand) - target)
        if gap < best_gap:
            best, best_gap = cand, gap
    assert best is not None
    return best


def build(kind: str, dims: dict[str, int], n_labels: int, **kw) -> nn.Module:
    """Construct a fusion model by name.

    `tri-can:<anchor>` selects the query modality, e.g. `tri-can:visual`.
    Plain `tri-can` keeps the historical default of anchoring on text.
    """
    if kind.startswith("hybrid-gated:"):
        return HybridFusion(dims, n_labels, anchor=kind.split(":", 1)[1],
                            gate=True, **kw)
    if kind == "hybrid-gated":
        return HybridFusion(dims, n_labels, gate=True, **kw)
    if kind.startswith("hybrid:"):
        return HybridFusion(dims, n_labels, anchor=kind.split(":", 1)[1], **kw)
    if kind == "hybrid":
        return HybridFusion(dims, n_labels, **kw)
    if kind.startswith("tri-can:"):
        return TriCAN(dims, n_labels, anchor=kind.split(":", 1)[1], **kw)
    if kind == "tri-can":
        return TriCAN(dims, n_labels, **kw)
    if kind == "late-fusion":
        return LateFusion(dims, n_labels, **kw)
    if kind == "gated-fusion":
        return GatedFusion(dims, n_labels,
                           **{k: v for k, v in kw.items() if k != "n_heads"})
    if kind == "all-pairs":
        return AllPairsCrossAttention(dims, n_labels, **kw)
    if kind == "early-fusion":
        return EarlyFusionTransformer(dims, n_labels, **kw)
    if kind == "late-fusion-wide":
        ref = TriCAN(dims, n_labels, **{k: v for k, v in kw.items() if k != "anchor"})
        return match_capacity(dims, n_labels, count_params(ref),
                              dropout=kw.get("dropout", 0.3))
    raise KeyError(f"unknown model {kind!r}")


# The comparison set. Ordered from least to most structured, so a table built
# in this order reads as "does added structure buy anything?".
MODELS = ("late-fusion", "late-fusion-wide", "gated-fusion",
          "tri-can:text", "tri-can:audio", "tri-can:visual",
          "hybrid:text", "hybrid:audio", "hybrid:visual", "hybrid-gated:text",
          "all-pairs", "early-fusion")
