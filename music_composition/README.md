# PIEC — Automatic Music Composition with an Evolutionary Algorithm

A from-scratch, dependency-light (numpy only) reproduction of

> **C.-K. Ting, C.-L. Wu, C.-H. Liu, "A Novel Automatic Composition System
> Using Evolutionary Algorithm and Phrase Imitation," IEEE Systems Journal, 2015.**

The course title *"Automatic Music Composition Using Evolutionary Algorithm
Based on Music Theory"* refers to this paper — its abstract literally
describes generating compositions *"by an evolutionary algorithm based on
music theory and imitation of … human-composed music."* The system is called
**PIEC** (Phrase-Imitation-based Evolutionary Composition).

Run it and you get a 40-second piece (WAV + MIDI) in `output/`.

```bash
pip install numpy
python3 compose.py                       # hybrid fitness (best)
python3 compose.py --fitness nv          # note-distribution only
python3 compose.py --fitness ra --gens 400
```

---

## 1. The idea of the paper

Pure rule-based evolutionary composition sounds *"machine-made"* because its
melodic progression is unpredictable; interactive GA (a human scores every
candidate) is accurate but suffers from listener fatigue. PIEC's insight:
**imitate the phrase-level behaviour of a human-composed sample melody** so the
result inherits a natural melodic shape, while a GA + music-theory fitness keeps
it musically valid.

PIEC takes a **sample melody** and keeps its **rhythm, chords and scale**,
then evolves the **pitches**. It imitates three characteristics of the sample:
**phrase motion** (ascending/descending contour), **note distribution** and
**interval distribution**, and it reproduces the sample's **repetition
structure**.

## 2. This reproduction — pipeline

```
sample melody (rhythm+chords+scale+phrase form fixed)
        │
        ▼
   GA population of pitch sequences
        │  selection (tournament)
        │  phrase-level crossover
        │  mutation (nearby scale tones)
        │  intra-phrase rearrangement   ← imitate phrase contour
        │  inter-phrase rearrangement   ← best phrase replaces its repeats
        │  fitness = NV / IV / RA / hybrid
        ▼
   best genome ── note fixing ──► render WAV + MIDI
```

Files:

| file | role |
|------|------|
| `theory.py`  | scale, diatonic triads, snap-to-scale / snap-to-chord helpers |
| `sample.py`  | the 16-bar C-major **A A B A** sample melody + per-bar chords |
| `piec.py`    | fitness functions, GA, phrase imitation, note fixing |
| `synth.py`   | numpy additive-synthesis WAV + hand-written MIDI |
| `compose.py` | CLI that runs the whole thing |

## 3. Representation

The sample supplies the fixed **rhythmic skeleton** (each note's onset and
duration), the **chord of every bar**, the **scale** (C major), and the
**phrase form** (`A A B A`, where the three `A` phrases are identical). The
**genome** is just the list of MIDI pitches, one per note event; the GA only
moves pitches. Pitches live on the C-major scale between C4 and C6.

## 4. The four fitness functions

Each returns a value in `[0, 1]` (higher is better). The paper's exact
formulas are paywalled, so these are faithful reconstructions of the described
behaviour (see *Reconstruction notes*).

- **NV — note-distribution similarity.** Duration-weighted pitch-class
  histogram of the candidate vs. the sample; `NV = 1 − ½·‖h_c − h_s‖₁`.
- **IV — interval-variance imitation.** Variance of within-phrase intervals vs.
  the sample's; `IV = 1 / (1 + |v_c − v_s|/8)` (a smooth falloff that keeps
  guiding the search instead of saturating to 0 for wild melodies).
- **RA — rule appropriateness.** Mean of **five music-theory rules**:
  1. **R1 scale membership** — every note is in the scale.
  2. **R2 chord tone on strong beats** — beats 1 & 3 land on chord tones.
  3. **R3 melodic smoothness** — graded per interval: step (≤M2) = 1.0,
     small skip (m3/M3) = 0.8, leap ≤ P5 = 0.4, ≥ m6 = 0.0, repeat = 0.3.
  4. **R4 leap resolution** — a leap (≥ P4) resolves by step in the opposite
     direction.
  5. **R5 cadence** — phrases end on chord tones and the piece ends on the
     tonic.
- **hybrid = 0.5·NV + 0.5·RA** (paper's best). The CLI default nudges in a
  little IV too (`0.35 NV + 0.40 RA + 0.25 IV`) for smoother contours.

## 5. Phrase imitation

- **Intra-phrase rearrangement.** Reorder a phrase's *own* pitches so its
  up/down contour matches the corresponding sample phrase (note multiset is
  preserved, motion is imitated).
- **Inter-phrase rearrangement.** Phrases sharing a label (the repeated `A`s)
  are detected; the highest-scoring generated phrase is copied over its
  repeats — reproducing the sample's repetition.

## 6. Note fixing

Applied during evolution and as a final polish: out-of-scale notes snap to the
nearest scale tone; strong-beat non-chord tones snap to a chord tone; every
phrase ends on a chord tone; the final note snaps to the **tonic** (authentic
cadence).

## 7. Results — the four fitness functions compared

Same sample, same seed, 300 generations, population 200:

| fitness | NV | RA | interval variance (sample = 10.6) | character |
|---------|----|----|-----------------------------------|-----------|
| **hybrid** | **0.94** | **0.94** | **9.3** | like the sample *and* rule-abiding — best |
| NV only | 0.81 | 0.70 | 70.9 | right pitch classes, but leaps everywhere |
| RA only | 0.80 | **0.98** | 25.1 | textbook-correct, less like the sample |
| IV only | 0.80 | 0.86 | 10.8 | interval shape closest, weaker on rules |

This matches the paper's conclusion that the **hybrid** objective produces the
most satisfying compositions. Audio for each is in `output/piec_<fitness>.wav`.

## 8. Reconstruction notes (honesty about the paper)

The full text (NTHU, IEEE Xplore, ResearchGate, Semantic Scholar) is blocked by
this environment's network policy, so the following were reconstructed from
abstracts, indexed snippets and standard music theory rather than copied from
the paper:

- the exact NV distance metric and IV formula;
- the precise wording of the five RA rules and their weighting;
- GA hyper-parameters (population, generations, rates);
- the mechanical details of intra-/inter-phrase rearrangement.

They are designed to match the *described behaviour* and produce musically
coherent output. Given the paper PDF, each can be tightened to the original.

## 9. Related work by the same group (for a report)

- C.-H. Liu, C.-K. Ting, *Evolutionary composition using music theory and
  charts*, IEEE CICAC/SSCI 2013 — the rule-based fitness predecessor.
- C.-H. Liu, C.-K. Ting, *Computational Intelligence in Music Composition: A
  Survey*, IEEE TETCI 1(1):2–15, 2017 — taxonomy of CI composition methods.
