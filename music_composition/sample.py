"""The sample (reference) melody that PIEC imitates.

Following the paper, PIEC *keeps the rhythm, chords and scale* of a sample
melody and evolves the pitches.  So the sample supplies:

  * the rhythmic skeleton (note onsets + durations)          -> fixed
  * the chord of every bar                                   -> fixed
  * the scale                                                -> fixed (C major)
  * the phrase structure incl. which phrases repeat          -> imitation target
  * the note-distribution / interval / contour statistics    -> fitness targets

The tune below is an original 16-bar melody in C major, 4/4, with an A A B A
phrase form (the three A phrases are identical, giving repeated-phrase
imitation something to lock onto).  Durations are in sixteenth notes; every
bar sums to 16.
"""

from __future__ import annotations

from dataclasses import dataclass

from theory import CHORDS

BAR_LEN = 16      # sixteenths per 4/4 bar
STRONG_POSITIONS = frozenset({0, 8})   # beats 1 and 3


# (pitch, duration_in_sixteenths) per bar --------------------------------------

_PHRASE_A = [
    [(64, 4), (67, 4), (72, 8)],            # I  (C):  E  G  C-
    [(69, 4), (67, 4), (65, 8)],            # IV (F):  A  G  F-
    [(67, 4), (71, 4), (74, 4), (71, 4)],   # V  (G):  G  B  D  B
    [(72, 4), (67, 4), (64, 8)],            # I  (C):  C  G  E-
]
_PHRASE_A_CHORDS = ["I", "IV", "V", "I"]

_PHRASE_B = [
    [(69, 4), (72, 4), (76, 8)],            # vi (Am): A  C  E-
    [(77, 4), (74, 4), (72, 4), (69, 4)],   # IV (F):  F  D  C  A
    [(74, 4), (72, 4), (69, 8)],            # ii (Dm): D  C  A-
    [(71, 4), (74, 4), (79, 4), (74, 4)],   # V  (G):  B  D  G  D
]
_PHRASE_B_CHORDS = ["vi", "IV", "ii", "V"]

# A A B A form.
_FORM = [
    ("A", _PHRASE_A, _PHRASE_A_CHORDS),
    ("A", _PHRASE_A, _PHRASE_A_CHORDS),
    ("B", _PHRASE_B, _PHRASE_B_CHORDS),
    ("A", _PHRASE_A, _PHRASE_A_CHORDS),
]


@dataclass
class Note:
    pitch: int          # MIDI number (mutable gene)
    start: int          # onset in sixteenths from piece start
    dur: int            # duration in sixteenths
    bar: int
    phrase: int         # phrase index 0..3
    pos: int            # position within the bar, in sixteenths
    chord: str          # roman numeral label
    strong: bool        # falls on a strong beat?

    @property
    def chord_pcs(self):
        return CHORDS[self.chord]


@dataclass
class Piece:
    notes: list[Note]
    phrase_labels: list[str]          # e.g. ['A','A','B','A']
    phrase_notes: list[list[int]]     # phrase idx -> list of note indices
    bar_chords: list[str]             # bar idx -> chord label
    n_bars: int

    def pitches(self):
        return [n.pitch for n in self.notes]

    def clone_pitches(self):
        return [n.pitch for n in self.notes]

    def set_pitches(self, pitches):
        for n, p in zip(self.notes, pitches):
            n.pitch = int(p)


def build_sample() -> Piece:
    notes: list[Note] = []
    phrase_labels: list[str] = []
    phrase_notes: list[list[int]] = []
    bar_chords: list[str] = []

    bar_idx = 0
    t = 0
    for phrase_idx, (label, bars, chords) in enumerate(_FORM):
        phrase_labels.append(label)
        idxs: list[int] = []
        for bar_in_phrase, (bar_events, chord) in enumerate(zip(bars, chords)):
            bar_chords.append(chord)
            pos = 0
            for pitch, dur in bar_events:
                idxs.append(len(notes))
                notes.append(Note(
                    pitch=pitch, start=t, dur=dur, bar=bar_idx,
                    phrase=phrase_idx, pos=pos, chord=chord,
                    strong=(pos in STRONG_POSITIONS),
                ))
                pos += dur
                t += dur
            assert pos == BAR_LEN, f"bar {bar_idx} sums to {pos}, not {BAR_LEN}"
            bar_idx += 1
        phrase_notes.append(idxs)

    return Piece(
        notes=notes,
        phrase_labels=phrase_labels,
        phrase_notes=phrase_notes,
        bar_chords=bar_chords,
        n_bars=bar_idx,
    )


if __name__ == "__main__":
    p = build_sample()
    print(f"bars={p.n_bars} notes={len(p.notes)} phrases={p.phrase_labels}")
    for ph, idxs in enumerate(p.phrase_notes):
        seq = [p.notes[i].pitch for i in idxs]
        print(f"  phrase {ph} [{p.phrase_labels[ph]}] chords="
              f"{[p.notes[i].chord for i in idxs]}")
        print(f"           pitches={seq}")
