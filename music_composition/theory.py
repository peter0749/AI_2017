"""Music-theory primitives for the PIEC reproduction.

Everything is expressed in MIDI note numbers (C4 = 60) and pitch classes
(0..11, where C = 0). The whole system works in C major / A minor so the
diatonic scale is fixed; changing KEY_ROOT is enough to transpose.
"""

from __future__ import annotations

# ---- Scale -----------------------------------------------------------------

KEY_ROOT = 0  # C

# Major scale pitch classes relative to the root.
MAJOR_STEPS = (0, 2, 4, 5, 7, 9, 11)
SCALE_PCS = frozenset((KEY_ROOT + s) % 12 for s in MAJOR_STEPS)

# Usable melodic range: C4 .. C6 restricted to scale tones.
MELODY_LO, MELODY_HI = 60, 84
SCALE_NOTES = tuple(p for p in range(MELODY_LO, MELODY_HI + 1) if p % 12 in SCALE_PCS)


# ---- Diatonic triads -------------------------------------------------------
# Roman-numeral -> triad pitch-class set, built by stacking thirds on the scale.

def _triad(degree_index: int) -> frozenset[int]:
    root = MAJOR_STEPS[degree_index % 7]
    third = MAJOR_STEPS[(degree_index + 2) % 7]
    fifth = MAJOR_STEPS[(degree_index + 4) % 7]
    return frozenset((KEY_ROOT + x) % 12 for x in (root, third, fifth))


CHORDS = {
    "I": _triad(0),    # C  E  G
    "ii": _triad(1),   # D  F  A
    "iii": _triad(2),  # E  G  B
    "IV": _triad(3),   # F  A  C
    "V": _triad(4),    # G  B  D
    "vi": _triad(5),   # A  C  E
    "vii": _triad(6),  # B  D  F
}

# Root pitch class of each chord, for building accompaniment / bass.
CHORD_ROOT_PC = {
    "I": 0, "ii": 2, "iii": 4, "IV": 5, "V": 7, "vi": 9, "vii": 11,
}

TONIC_PC = KEY_ROOT  # ending target


# ---- Helpers ---------------------------------------------------------------

def nearest_scale_note(pitch: int) -> int:
    """Snap an arbitrary MIDI pitch to the closest in-scale note in range."""
    pitch = max(MELODY_LO, min(MELODY_HI, pitch))
    return min(SCALE_NOTES, key=lambda s: (abs(s - pitch), s))


def nearest_chord_note(pitch: int, chord_pcs: frozenset[int]) -> int:
    """Snap a MIDI pitch to the closest in-range note whose pitch class is a
    chord tone of ``chord_pcs``."""
    pitch = max(MELODY_LO, min(MELODY_HI, pitch))
    candidates = [p for p in range(MELODY_LO, MELODY_HI + 1) if p % 12 in chord_pcs]
    return min(candidates, key=lambda s: (abs(s - pitch), s))


def is_scale_tone(pitch: int) -> bool:
    return pitch % 12 in SCALE_PCS


def is_chord_tone(pitch: int, chord_pcs: frozenset[int]) -> bool:
    return pitch % 12 in chord_pcs
