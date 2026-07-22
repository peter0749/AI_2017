"""PIEC - Phrase-Imitation-based Evolutionary Composition (reproduction).

Reproduction of the method in

    C.-K. Ting, C.-L. Wu, C.-H. Liu,
    "A Novel Automatic Composition System Using Evolutionary Algorithm and
     Phrase Imitation", IEEE Systems Journal, 2015.

The paper's exact fitness equations are behind a paywall, so the four fitness
functions here are faithful reconstructions of the *described* behaviour:

    NV      note-distribution similarity to the sample melody
    IV      imitation of the sample's interval variance (spread)
    RA      rule appropriateness - five music-theory rules
    hybrid  0.5 * NV + 0.5 * RA   (the paper's best-performing combination)

On top of the GA the system performs, exactly as described in the paper:
    * intra-phrase rearrangement  - imitate a phrase's ascending/descending
                                    contour using the candidate's own notes
    * inter-phrase rearrangement  - detect repeated phrases and let the best
                                    generated phrase replace its repeats
    * note fixing                 - snap out-of-scale / out-of-chord notes and
                                    force a tonic cadence, during & after runs
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from sample import Piece
from theory import (
    SCALE_NOTES, TONIC_PC, MELODY_LO, MELODY_HI,
    nearest_scale_note, nearest_chord_note, is_scale_tone, is_chord_tone,
)

Genome = np.ndarray  # int array, one MIDI pitch per note event


# ---------------------------------------------------------------------------
# Statistics used by the fitness functions
# ---------------------------------------------------------------------------

def _smoothness(abs_interval: float) -> float:
    """Per-interval melodic-smoothness score in [0, 1]."""
    a = abs(abs_interval)
    if a == 0:
        return 0.3          # repeated note: acceptable but static
    if a <= 2:
        return 1.0          # step (m2/M2): ideal
    if a <= 4:
        return 0.8          # small skip (m3/M3)
    if a <= 7:
        return 0.4          # leap up to a fifth
    return 0.0              # sixth or wider: avoid


def pc_histogram(pitches, piece: Piece) -> np.ndarray:
    """Duration-weighted pitch-class distribution (12 bins, sums to 1)."""
    h = np.zeros(12)
    for p, n in zip(pitches, piece.notes):
        h[int(p) % 12] += n.dur
    s = h.sum()
    return h / s if s > 0 else h


def phrase_intervals(pitches, piece: Piece) -> np.ndarray:
    """Signed semitone intervals between consecutive notes, within phrases."""
    ivs = []
    for idxs in piece.phrase_notes:
        seq = [int(pitches[i]) for i in idxs]
        ivs.extend(np.diff(seq).tolist())
    return np.asarray(ivs, dtype=float)


# ---------------------------------------------------------------------------
# Evaluator: holds the sample-derived targets and scores candidate genomes
# ---------------------------------------------------------------------------

@dataclass
class Weights:
    nv: float = 0.5
    ra: float = 0.5
    iv: float = 0.0    # folded into `hybrid` only if > 0


class Evaluator:
    IV_TAU = 4.0        # interval-variance tolerance
    LEAP = 5            # >= perfect fourth counts as a leap
    STEP = 2            # <= major second counts as a step

    def __init__(self, piece: Piece):
        self.piece = piece
        sample = np.asarray(piece.pitches())
        self.sample_hist = pc_histogram(sample, piece)
        self.sample_ivar = float(np.var(phrase_intervals(sample, piece)))

    # ---- the four fitness functions (each returns a value in [0, 1]) ----

    def nv(self, g: Genome) -> float:
        h = pc_histogram(g, self.piece)
        l1 = float(np.abs(h - self.sample_hist).sum())   # in [0, 2]
        return 1.0 - 0.5 * l1

    def iv(self, g: Genome) -> float:
        # Gradient-preserving similarity to the sample's interval spread: a
        # plain exp() saturates to 0 for wild melodies and stops guiding the
        # search, so use a smooth rational falloff that is never fully flat.
        v = float(np.var(phrase_intervals(g, self.piece)))
        return 1.0 / (1.0 + abs(v - self.sample_ivar) / 8.0)

    def ra(self, g: Genome) -> float:
        return float(np.mean(self._ra_rules(g)))

    def hybrid(self, g: Genome, w: Weights) -> float:
        val = w.nv * self.nv(g) + w.ra * self.ra(g)
        if w.iv > 0:
            val += w.iv * self.iv(g)
        return val / (w.nv + w.ra + w.iv)

    # ---- the five music-theory rules behind RA -------------------------

    def _ra_rules(self, g: Genome) -> list[float]:
        notes = self.piece.notes

        # R1  every note should belong to the scale
        r1 = np.mean([is_scale_tone(int(p)) for p in g])

        # R2  notes on strong beats should be chord tones
        strong = [(int(p), n) for p, n in zip(g, notes) if n.strong]
        r2 = (np.mean([is_chord_tone(p, n.chord_pcs) for p, n in strong])
              if strong else 1.0)

        ivs = phrase_intervals(g, self.piece)
        aiv = np.abs(ivs)

        # R3  melodic smoothness: graded reward per interval - steps are best,
        # small skips good, leaps discouraged, octave+ leaps forbidden.
        r3 = float(np.mean([_smoothness(x) for x in aiv])) if len(ivs) else 0.5

        # R4  large leaps should resolve by step in the opposite direction
        leaps = np.where(aiv >= self.LEAP)[0]
        if len(leaps) == 0:
            r4 = 1.0
        else:
            ok = 0
            for i in leaps:
                if i + 1 < len(ivs):
                    opp = np.sign(ivs[i]) != np.sign(ivs[i + 1])
                    small = abs(ivs[i + 1]) <= self.STEP
                    ok += bool(opp and small)
            r4 = ok / len(leaps)

        # R5  cadence: phrases end on chord tones, whole piece ends on tonic
        phrase_end_ok = []
        for idxs in self.piece.phrase_notes:
            last = notes[idxs[-1]]
            phrase_end_ok.append(is_chord_tone(int(g[idxs[-1]]), last.chord_pcs))
        final_tonic = 1.0 if int(g[-1]) % 12 == TONIC_PC else (
            0.5 if is_chord_tone(int(g[-1]), notes[-1].chord_pcs) else 0.0)
        r5 = 0.5 * np.mean(phrase_end_ok) + 0.5 * final_tonic

        return [float(r1), float(r2), float(r3), float(r4), float(r5)]

    def phrase_ra(self, g: Genome, phrase_idx: int) -> float:
        """Cheap RA-like score of a single phrase, used to pick the best copy
        for inter-phrase rearrangement."""
        idxs = self.piece.phrase_notes[phrase_idx]
        seq = [int(g[i]) for i in idxs]
        notes = [self.piece.notes[i] for i in idxs]
        scale = np.mean([is_scale_tone(p) for p in seq])
        strong = [(p, n) for p, n in zip(seq, notes) if n.strong]
        chord = np.mean([is_chord_tone(p, n.chord_pcs) for p, n in strong]) if strong else 1.0
        aiv = np.abs(np.diff(seq)) if len(seq) > 1 else np.array([0])
        smooth = np.mean((aiv >= 1) & (aiv <= 4))
        end = is_chord_tone(seq[-1], notes[-1].chord_pcs)
        return float(0.3 * scale + 0.3 * chord + 0.2 * smooth + 0.2 * end)


# ---------------------------------------------------------------------------
# Phrase imitation  (intra-phrase & inter-phrase rearrangement)
# ---------------------------------------------------------------------------

def intraphrase_imitate(g: Genome, piece: Piece, sample: np.ndarray, phrase_idx: int):
    """Reorder a phrase's *own* pitches so its up/down contour matches the
    corresponding sample phrase (note multiset preserved, motion imitated)."""
    idxs = piece.phrase_notes[phrase_idx]
    if len(idxs) < 2:
        return
    samp = np.asarray([sample[i] for i in idxs])
    cand = np.asarray([g[i] for i in idxs])
    order = np.argsort(samp, kind="stable")     # positions low->high in sample
    ranked = np.sort(cand, kind="stable")       # candidate pitches low->high
    new = np.empty_like(cand)
    new[order] = ranked
    for k, gi in enumerate(idxs):
        g[gi] = int(new[k])


def interphrase_imitate(g: Genome, piece: Piece, ev: Evaluator):
    """Detect repeated phrases (same label) and copy the best-scoring one over
    its repeats -- imitating the repetition structure of the sample."""
    groups: dict[str, list[int]] = defaultdict(list)
    for pi, lab in enumerate(piece.phrase_labels):
        groups[lab].append(pi)
    for lab, phrases in groups.items():
        if len(phrases) < 2:
            continue
        best = max(phrases, key=lambda pi: ev.phrase_ra(g, pi))
        best_pitches = [int(g[i]) for i in piece.phrase_notes[best]]
        for pi in phrases:
            if pi == best:
                continue
            for k, gi in enumerate(piece.phrase_notes[pi]):
                g[gi] = best_pitches[k]


# ---------------------------------------------------------------------------
# Note fixing
# ---------------------------------------------------------------------------

def _nearest_tonic(pitch: int) -> int:
    pitch = max(MELODY_LO, min(MELODY_HI, pitch))
    toniccs = [p for p in range(MELODY_LO, MELODY_HI + 1) if p % 12 == TONIC_PC]
    return min(toniccs, key=lambda s: (abs(s - pitch), s))


def note_fix(g: Genome, piece: Piece):
    """Snap out-of-scale notes to the scale, out-of-chord strong-beat notes to
    a chord tone, phrase endings to chord tones, and the very last note to the
    tonic (authentic cadence)."""
    for i, n in enumerate(piece.notes):
        p = int(g[i])
        if not is_scale_tone(p):
            p = nearest_scale_note(p)
        if n.strong and not is_chord_tone(p, n.chord_pcs):
            p = nearest_chord_note(p, n.chord_pcs)
        g[i] = p
    for idxs in piece.phrase_notes:
        j = idxs[-1]
        nj = piece.notes[j]
        if not is_chord_tone(int(g[j]), nj.chord_pcs):
            g[j] = nearest_chord_note(int(g[j]), nj.chord_pcs)
    g[-1] = _nearest_tonic(int(g[-1]))


# ---------------------------------------------------------------------------
# The genetic algorithm
# ---------------------------------------------------------------------------

@dataclass
class Config:
    pop_size: int = 200
    generations: int = 300
    tournament: int = 3
    p_crossover: float = 0.9      # phrase-level uniform crossover
    p_mutate: float = 0.06        # per-gene
    p_intra: float = 0.5          # apply intra-phrase imitation to offspring
    p_inter: float = 0.3          # apply inter-phrase imitation to offspring
    elitism: int = 4
    fitness: str = "hybrid"       # nv | iv | ra | hybrid
    weights: Weights = field(default_factory=Weights)
    seed: int = 20170721


class PIEC:
    def __init__(self, piece: Piece, cfg: Config):
        self.piece = piece
        self.cfg = cfg
        self.ev = Evaluator(piece)
        self.rng = np.random.default_rng(cfg.seed)
        self.sample = np.asarray(piece.pitches())
        self.n = len(piece.notes)
        self._score = {
            "nv": self.ev.nv,
            "iv": self.ev.iv,
            "ra": self.ev.ra,
            "hybrid": lambda g: self.ev.hybrid(g, cfg.weights),
        }[cfg.fitness]

    # ---- population helpers ----

    def _random_genome(self) -> Genome:
        return self.rng.choice(SCALE_NOTES, size=self.n).astype(int)

    def _mutate(self, g: Genome):
        for i in range(self.n):
            if self.rng.random() < self.cfg.p_mutate:
                idx = SCALE_NOTES.index(nearest_scale_note(int(g[i])))
                step = int(self.rng.integers(-2, 3))         # -2..+2 scale steps
                idx = min(max(idx + step, 0), len(SCALE_NOTES) - 1)
                g[i] = SCALE_NOTES[idx]

    def _crossover(self, a: Genome, b: Genome) -> Genome:
        """Phrase-level uniform crossover: each phrase comes wholesale from one
        parent, keeping phrases intact."""
        child = a.copy()
        if self.rng.random() > self.cfg.p_crossover:
            return child
        for idxs in self.piece.phrase_notes:
            src = a if self.rng.random() < 0.5 else b
            for gi in idxs:
                child[gi] = src[gi]
        return child

    def _select(self, pop, fits) -> Genome:
        k = self.cfg.tournament
        pick = self.rng.integers(0, len(pop), size=k)
        best = pick[int(np.argmax(fits[pick]))]
        return pop[best].copy()

    # ---- main loop ----

    def run(self, verbose: bool = True):
        cfg = self.cfg
        pop = [self._random_genome() for _ in range(cfg.pop_size)]
        fits = np.array([self._score(g) for g in pop])
        history = []

        for gen in range(cfg.generations):
            order = np.argsort(-fits)
            elites = [pop[i].copy() for i in order[:cfg.elitism]]
            new_pop = elites[:]

            while len(new_pop) < cfg.pop_size:
                p1 = self._select(pop, fits)
                p2 = self._select(pop, fits)
                child = self._crossover(p1, p2)
                self._mutate(child)
                if self.rng.random() < cfg.p_intra:
                    ph = int(self.rng.integers(0, len(self.piece.phrase_notes)))
                    intraphrase_imitate(child, self.piece, self.sample, ph)
                if self.rng.random() < cfg.p_inter:
                    interphrase_imitate(child, self.piece, self.ev)
                new_pop.append(child)

            pop = new_pop
            fits = np.array([self._score(g) for g in pop])
            best = float(fits.max())
            history.append(best)
            if verbose and (gen % 25 == 0 or gen == cfg.generations - 1):
                bi = int(np.argmax(fits))
                r = self.ev._ra_rules(pop[bi])
                print(f"gen {gen:3d}  best={best:.3f}  "
                      f"NV={self.ev.nv(pop[bi]):.2f} RA={self.ev.ra(pop[bi]):.2f} "
                      f"IV={self.ev.iv(pop[bi]):.2f}  rules={[round(x,2) for x in r]}")

        best_g = pop[int(np.argmax(fits))].copy()
        note_fix(best_g, self.piece)          # final polish
        return best_g, history

    # ---- reporting ----

    def report(self, g: Genome) -> dict:
        r = self.ev._ra_rules(g)
        return {
            "NV": round(self.ev.nv(g), 3),
            "IV": round(self.ev.iv(g), 3),
            "RA": round(self.ev.ra(g), 3),
            "hybrid": round(self.ev.hybrid(g, self.cfg.weights), 3),
            "rules(R1..R5)": [round(x, 3) for x in r],
            "sample_ivar": round(self.ev.sample_ivar, 2),
            "cand_ivar": round(float(np.var(phrase_intervals(g, self.piece))), 2),
        }


if __name__ == "__main__":
    from sample import build_sample
    piece = build_sample()
    cfg = Config(fitness="hybrid", weights=Weights(nv=0.35, ra=0.40, iv=0.25))
    piec = PIEC(piece, cfg)
    best, hist = piec.run()
    print("\nfinal metrics:", piec.report(best))
    print("final pitches:", best.tolist())
