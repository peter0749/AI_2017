"""Turn an evolved melody into something you can actually listen to.

Two outputs, no external audio libraries required:

  * a 16-bit PCM WAV  (numpy synthesis: melody + soft chord pad + bass)
  * a Standard MIDI File (hand-written bytes, so it opens in any DAW)

The synth aims for a *warm, mellow* tone: a couple of gently-detuned
oscillators with only a few soft harmonics, click-free raised-cosine
envelopes, and a smooth FFT-convolution reverb (not slap-back echoes).
Pick a timbre with ``instrument=`` : 'soft' (default), 'epiano', 'flute'.
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
# Instrument presets
# ---------------------------------------------------------------------------
# harm   : relative amplitudes of harmonics 1..k (fewer, softer = mellower)
# env    : 'adsr' (sustained) or 'pluck' (exponential decay)
# detune : cents of spread between the two oscillators (warmth)
# vib    : vibrato depth (delayed, subtle)

INSTRUMENTS = {
    "soft":   dict(harm=[1.0, 0.22, 0.08, 0.02], env="adsr",
                   a=0.014, d=0.13, s=0.55, r=0.22, detune=5.0, vib=0.0),
    "epiano": dict(harm=[1.0, 0.32, 0.11, 0.03], env="pluck",
                   a=0.005, decay=2.2, r=0.10, detune=4.0, vib=0.0),
    "flute":  dict(harm=[1.0, 0.05, 0.02], env="adsr",
                   a=0.055, d=0.10, s=0.82, r=0.16, detune=0.0, vib=0.0),
}


def _cos_ramp(n: int) -> np.ndarray:
    """Raised-cosine ramp 0->1 (click-free)."""
    if n <= 0:
        return np.ones(0)
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def _env_adsr(n: int, a, d, s, r) -> np.ndarray:
    env = np.full(n, s, dtype=float)
    na, nd, nr = min(int(a * SR), n), min(int(d * SR), n), min(int(r * SR), n)
    if na:
        env[:na] = _cos_ramp(na)
    if nd and na + nd <= n:
        env[na:na + nd] = 1.0 + (s - 1.0) * _cos_ramp(nd)
    if nr:
        env[n - nr:] *= _cos_ramp(nr)[::-1]
    return env


def _env_pluck(n: int, a, decay, r) -> np.ndarray:
    t = np.arange(n) / SR
    env = np.exp(-decay * t)
    na, nr = min(int(a * SR), n), min(int(r * SR), n)
    if na:
        env[:na] *= _cos_ramp(na)
    if nr:
        env[n - nr:] *= _cos_ramp(nr)[::-1]
    return env


def _voice(freq: float, dur_s: float, preset: dict, gain: float) -> np.ndarray:
    n = int(dur_s * SR)
    if n <= 0:
        return np.zeros(0)
    t = np.arange(n) / SR

    # delayed, subtle vibrato
    vib = preset["vib"]
    if vib:
        onset = np.clip((t - 0.15) / 0.2, 0, 1)
        fmod = 1.0 + vib * onset * np.sin(2 * np.pi * 5.2 * t)
    else:
        fmod = 1.0

    det = preset["detune"] / 1200.0
    tone = np.zeros(n)
    for osc, ratio in ((1.0, 2 ** det), (1.0, 2 ** -det)):
        phase = 2 * np.pi * freq * ratio * fmod * t
        for k, amp in enumerate(preset["harm"], start=1):
            tone += osc * amp * np.sin(k * phase)
    tone /= 2 * sum(preset["harm"])            # normalise before enveloping

    if preset["env"] == "pluck":
        env = _env_pluck(n, preset["a"], preset["decay"], preset["r"])
    else:
        env = _env_adsr(n, preset["a"], preset["d"], preset["s"], preset["r"])
    return gain * tone * env


def _place(buf: np.ndarray, sig: np.ndarray, start_s: float):
    i = int(start_s * SR)
    j = min(i + len(sig), len(buf))
    if i < len(buf):
        buf[i:j] += sig[:j - i]


# ---------------------------------------------------------------------------
# Smooth reverb via FFT convolution with a decaying, low-passed noise IR
# ---------------------------------------------------------------------------

def _reverb(buf: np.ndarray, wet=0.22, decay_s=0.5, length_s=0.7) -> np.ndarray:
    n_ir = int(length_s * SR)
    rng = np.random.default_rng(0)
    ir = rng.standard_normal(n_ir) * np.exp(-np.arange(n_ir) / (decay_s * SR))
    # gentle low-pass on the tail (moving average) so it isn't fizzy
    w = max(1, int(SR / 3500))
    ir = np.convolve(ir, np.ones(w) / w, mode="same")
    ir[: int(0.008 * SR)] = 0.0                # small pre-delay, keep dry attack
    ir /= np.abs(ir).sum() or 1.0

    n = len(buf) + n_ir - 1
    N = 1 << int(np.ceil(np.log2(n)))
    wetbuf = np.fft.irfft(np.fft.rfft(buf, N) * np.fft.rfft(ir, N), N)[:len(buf)]
    return (1 - wet) * buf + wet * wetbuf


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------

def render_wav(piece: Piece, pitches, path: str,
               with_accompaniment: bool = True, instrument: str = "soft"):
    preset = INSTRUMENTS.get(instrument, INSTRUMENTS["soft"])
    pad_preset = INSTRUMENTS["soft"]

    total_16ths = piece.n_bars * BAR_LEN
    buf = np.zeros(int((total_16ths * SEC_PER_16TH + 1.0) * SR))

    # --- melody -----------------------------------------------------------
    for p, n in zip(pitches, piece.notes):
        sig = _voice(midi_to_hz(int(p)), n.dur * SEC_PER_16TH + 0.06,
                     preset, gain=0.52)
        _place(buf, sig, n.start * SEC_PER_16TH)

    if with_accompaniment:
        for bar in range(piece.n_bars):
            chord = piece.bar_chords[bar]
            start_s = bar * BAR_LEN * SEC_PER_16TH
            dur_s = BAR_LEN * SEC_PER_16TH
            for pc in CHORDS[chord]:                       # soft triad pad (C3)
                sig = _voice(midi_to_hz(48 + pc), dur_s + 0.1,
                             pad_preset, gain=0.085)
                _place(buf, sig, start_s)
            for half in range(2):                          # bass root (C2)
                sig = _voice(midi_to_hz(36 + CHORD_ROOT_PC[chord]),
                             dur_s / 2 * 0.98, INSTRUMENTS["epiano"], gain=0.20)
                _place(buf, sig, start_s + half * dur_s / 2)

    buf = _reverb(buf)

    peak = np.max(np.abs(buf)) or 1.0
    buf = np.tanh(buf / peak * 1.1) * 0.9              # soft limiter, no clipping
    pcm = (buf / (np.max(np.abs(buf)) or 1.0) * 0.9 * 32767).astype(np.int16)
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
    uspq = int(60_000_000 / BPM)
    tempo = bytearray(_vlq(0)) + bytes([0xFF, 0x51, 0x03]) + uspq.to_bytes(3, "big")

    for p, n in zip(pitches, piece.notes):                # melody, channel 0
        events.append((n.start * TPS, 0x90, int(p), 92))
        events.append(((n.start + n.dur) * TPS, 0x80, int(p), 0))

    for bar in range(piece.n_bars):                       # chords ch1, bass ch2
        chord = piece.bar_chords[bar]
        on = bar * BAR_LEN * TPS
        off = (bar + 1) * BAR_LEN * TPS
        for pc in CHORDS[chord]:
            events.append((on, 0x91, 48 + pc, 55))
            events.append((off, 0x81, 48 + pc, 0))
        bass = 36 + CHORD_ROOT_PC[chord]
        events.append((on, 0x92, bass, 70))
        events.append((off, 0x82, bass, 0))

    body = bytearray(tempo)
    prev = 0
    for tick, status, d1, d2 in sorted(events, key=lambda e: (e[0], e[1] & 0xF0 == 0x90)):
        body += _vlq(tick - prev)
        body += bytes([status, d1, d2])
        prev = tick
    body += _vlq(0) + bytes([0xFF, 0x2F, 0x00])
    track = b"MTrk" + struct.pack(">I", len(body)) + bytes(body)

    header = b"MThd" + struct.pack(">IHHH", 6, 0, 1, DIV)
    with open(path, "wb") as f:
        f.write(header + track)
    return path
