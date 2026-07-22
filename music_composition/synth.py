"""Turn an evolved melody into something you can actually listen to.

Two outputs, no external audio libraries required:

  * a 16-bit PCM WAV  (numpy additive synthesis: melody + soft chord pad + bass)
  * a Standard MIDI File (hand-written bytes, so it opens in any DAW)
"""

from __future__ import annotations

import struct
import wave

import numpy as np

from sample import Piece, BAR_LEN
from theory import CHORD_ROOT_PC, CHORDS

SR = 44100          # sample rate
BPM = 96
SEC_PER_16TH = 60.0 / BPM / 4.0


def midi_to_hz(m: float) -> float:
    return 440.0 * 2.0 ** ((m - 69) / 12.0)


# ---------------------------------------------------------------------------
# WAV synthesis
# ---------------------------------------------------------------------------

def _adsr(n: int, a=0.01, d=0.10, s=0.7, r=0.12) -> np.ndarray:
    """Simple attack/decay/sustain/release envelope of length n samples."""
    env = np.ones(n)
    na, nd, nr = int(a * SR), int(d * SR), int(r * SR)
    na, nd, nr = min(na, n), min(nd, n), min(nr, n)
    if na:
        env[:na] = np.linspace(0, 1, na)
    if nd:
        env[na:na + nd] = np.linspace(1, s, nd)
    env[na + nd:n - nr] = s
    if nr:
        env[n - nr:] = np.linspace(env[n - nr - 1] if n - nr - 1 >= 0 else s, 0, nr)
    return env


def _voice(freq: float, dur_s: float, harmonics, gain: float,
           a=0.01, d=0.10, s=0.7, r=0.12, vibrato=0.0) -> np.ndarray:
    n = int(dur_s * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR
    wave_ = np.zeros(n)
    vib = 1.0 + vibrato * np.sin(2 * np.pi * 5.0 * t) if vibrato else 1.0
    for k, amp in enumerate(harmonics, start=1):
        wave_ += amp * np.sin(2 * np.pi * freq * k * t * vib)
    return gain * wave_ * _adsr(n, a, d, s, r)


def _reverb(buf: np.ndarray) -> np.ndarray:
    """Cheap multi-tap reverb for a bit of air around the dry synth tone."""
    out = buf.copy()
    for delay_ms, gain in [(47, 0.28), (89, 0.18), (131, 0.12), (173, 0.07)]:
        d = int(delay_ms / 1000 * SR)
        out[d:] += gain * buf[:-d]
    return out


def _place(buf: np.ndarray, sig: np.ndarray, start_s: float):
    i = int(start_s * SR)
    j = min(i + len(sig), len(buf))
    if i < len(buf):
        buf[i:j] += sig[:j - i]


def render_wav(piece: Piece, pitches, path: str,
               with_accompaniment: bool = True):
    total_16ths = piece.n_bars * BAR_LEN
    total_s = total_16ths * SEC_PER_16TH + 0.5
    buf = np.zeros(int(total_s * SR))

    # --- melody -----------------------------------------------------------
    mel_harm = [1.0, 0.45, 0.28, 0.16, 0.08]     # slightly reedy
    for p, n in zip(pitches, piece.notes):
        sig = _voice(midi_to_hz(int(p)), n.dur * SEC_PER_16TH,
                     mel_harm, gain=0.5, a=0.008, d=0.08, s=0.75, r=0.10,
                     vibrato=0.004)
        _place(buf, sig, n.start * SEC_PER_16TH)

    if with_accompaniment:
        pad_harm = [1.0, 0.5, 0.25]
        # --- chord pad + bass, one chord per bar --------------------------
        for bar in range(piece.n_bars):
            chord = piece.bar_chords[bar]
            root_pc = CHORD_ROOT_PC[chord]
            start_s = bar * BAR_LEN * SEC_PER_16TH
            dur_s = BAR_LEN * SEC_PER_16TH
            # soft triad pad around C3..B3
            for pc in CHORDS[chord]:
                midi = 48 + pc                    # C3 register
                sig = _voice(midi_to_hz(midi), dur_s, pad_harm, gain=0.10,
                             a=0.03, d=0.2, s=0.6, r=0.25)
                _place(buf, sig, start_s)
            # bass: root, two half-note pulses per bar
            for half in range(2):
                bass_midi = 36 + root_pc          # C2 register
                sig = _voice(midi_to_hz(bass_midi), dur_s / 2 * 0.95,
                             [1.0, 0.3], gain=0.22, a=0.01, d=0.1, s=0.5, r=0.1)
                _place(buf, sig, start_s + half * dur_s / 2)

    buf = _reverb(buf)

    # --- normalise & write 16-bit PCM ------------------------------------
    peak = np.max(np.abs(buf)) or 1.0
    buf = (buf / peak) * 0.89
    pcm = (buf * 32767).astype(np.int16)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return path


# ---------------------------------------------------------------------------
# Minimal Standard MIDI File writer
# ---------------------------------------------------------------------------

DIV = 480                         # ticks per quarter note
TPS = DIV // 4                    # ticks per sixteenth


def _vlq(n: int) -> bytes:
    """MIDI variable-length quantity."""
    out = bytearray([n & 0x7F])
    n >>= 7
    while n:
        out.insert(0, (n & 0x7F) | 0x80)
        n >>= 7
    return bytes(out)


def render_midi(piece: Piece, pitches, path: str):
    events = []
    # tempo meta on tick 0
    uspq = int(60_000_000 / BPM)
    tempo = bytearray(_vlq(0)) + bytes([0xFF, 0x51, 0x03]) + uspq.to_bytes(3, "big")

    # melody, channel 0
    for p, n in zip(pitches, piece.notes):
        on = n.start * TPS
        off = (n.start + n.dur) * TPS
        events.append((on, 0x90, int(p), 92))
        events.append((off, 0x80, int(p), 0))

    # chords, channel 1 ; bass, channel 2
    for bar in range(piece.n_bars):
        chord = piece.bar_chords[bar]
        on = bar * BAR_LEN * TPS
        off = (bar + 1) * BAR_LEN * TPS
        for pc in CHORDS[chord]:
            note = 48 + pc
            events.append((on, 0x91, note, 55))
            events.append((off, 0x81, note, 0))
        bass = 36 + CHORD_ROOT_PC[chord]
        events.append((on, 0x92, bass, 70))
        events.append((off, 0x82, bass, 0))

    # single track: tempo meta first, then all note events in delta time
    body = bytearray(tempo)
    prev = 0
    for tick, status, d1, d2 in sorted(events, key=lambda e: (e[0], e[1] & 0xF0 == 0x90)):
        body += _vlq(tick - prev)
        body += bytes([status, d1, d2])
        prev = tick
    body += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    track_chunk = b"MTrk" + struct.pack(">I", len(body)) + bytes(body)

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, DIV)
    with open(path, "wb") as f:
        f.write(header + track_chunk)
    return path
