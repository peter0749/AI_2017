"""End-to-end: evolve a melody with PIEC and render it to WAV + MIDI.

    python3 compose.py [--fitness hybrid|nv|iv|ra] [--gens N] [--pop N]
                       [--seed N] [--out NAME]
"""

from __future__ import annotations

import argparse
import os

from sample import build_sample
from piec import PIEC, Config, Weights
from synth import render_wav, render_midi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fitness", default="hybrid",
                    choices=["hybrid", "nv", "iv", "ra"])
    ap.add_argument("--gens", type=int, default=300)
    ap.add_argument("--pop", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20170721)
    ap.add_argument("--out", default="piec_composition")
    ap.add_argument("--no-accomp", action="store_true")
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(here, "output")
    os.makedirs(out_dir, exist_ok=True)

    piece = build_sample()
    cfg = Config(
        fitness=args.fitness, generations=args.gens, pop_size=args.pop,
        seed=args.seed, weights=Weights(nv=0.35, ra=0.40, iv=0.25),
    )

    print(f"== PIEC ({args.fitness}) : {piece.n_bars} bars, "
          f"{len(piece.notes)} notes, form {piece.phrase_labels} ==")
    piec = PIEC(piece, cfg)
    best, history = piec.run()

    print("\nfinal metrics:", piec.report(best))
    names = {0: "C", 1: "C#", 2: "D", 3: "D#", 4: "E", 5: "F", 6: "F#",
             7: "G", 8: "G#", 9: "A", 10: "A#", 11: "B"}
    spelled = [f"{names[p % 12]}{p // 12 - 1}" for p in best.tolist()]
    print("final melody:", " ".join(spelled))

    wav = os.path.join(out_dir, args.out + ".wav")
    mid = os.path.join(out_dir, args.out + ".mid")
    render_wav(piece, best, wav, with_accompaniment=not args.no_accomp)
    render_midi(piece, best, mid)
    print(f"\nwrote {wav}\nwrote {mid}")


if __name__ == "__main__":
    main()
