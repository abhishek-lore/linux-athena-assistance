#!/usr/bin/env python3
"""
Local, offline, GPU-accelerated voice assistant ("Nyx").

Pipeline:  mic -> openWakeWord (defaults to the pretrained "hey jarvis"
           model; swap in your own trained phrase via WAKE_WORD_MODEL_PATH
           below — see wake_word_training/README.md)
           -> VAD-gated recording -> faster-whisper (STT) -> Ollama (streamed)
           -> Piper (TTS, streamed sentence-by-sentence, interruptible)
           -> speakers

Everything runs on-device. No cloud calls, no API keys.

Features in this version:
  - Barge-in: say "Hey Jarvis" again *while Nyx is talking* and it stops
    immediately and starts listening. NOTE: barge-in only reacts to the
    wake word, not to arbitrary speech — reacting to any loud sound would
    mean Nyx's own TTS output (which is speech, picked up by the mic)
    would constantly interrupt itself. This deliberately avoids needing
    any PipeWire echo-cancellation setup.
  - Streaming LLM replies: Piper starts speaking each sentence as soon as
    Ollama produces it instead of waiting for the whole reply, so replies
    start playing much sooner.
  - VAD (webrtcvad) instead of a fixed RMS threshold for silence
    detection — adapts much better to mic gain / room noise than a
    hardcoded number.
  - Media tools: "stop", "pause", "resume", "skip"/"next", "volume
    up"/"volume down", all sent to mpv over its IPC socket (no PipeWire
    config needed).
  - System launch tools: "open Firefox", "open YouTube", "open Downloads"
    etc. — a small, explicit whitelist of apps/websites/folders matched
    by keyword, same closed-whitelist pattern as the media tools. Nyx
    never runs an arbitrary command; unrecognized phrases just fall
    through to a normal LLM reply.
  - Web search: "search for X" opens a Google search for X in the
    browser. Like play_music, the query is free-form text, but it's
    only ever URL-encoded into a search link and opened — never
    executed as a command.
  - Date/time: "what time is it" / "what's the date" answered directly
    from the system clock, no LLM call involved.
  - System toggles: "turn wifi on/off" (nmcli), "turn bluetooth on/off"
    (bluetoothctl), "turn night light on/off" (hyprsunset) — same
    closed-whitelist pattern as everything else here.
  - System volume: "volume up/down", "mute"/"unmute", "set volume to
    N percent" via wpctl (PipeWire) — affects the whole system, not
    just Nyx's own playback.
  - Screen brightness: "brightness up/down", "set brightness to N
    percent" via brightnessctl.
  - Persistent memory: conversation history is saved to disk and reloaded
    on the next run.
  - Overlay widget: pushes live state (idle/listening/thinking/speaking)
    and audio level to a Unix socket (see overlay_ipc.py) for the
    NyxOrb quickshell widget to render.
"""

import json
import os
import queue
import random
import re
import socket
import subprocess
import sys
import time
import urllib.parse
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import sounddevice as sd
import webrtcvad

import openwakeword
from openwakeword.model import Model as WakeWordModel
from faster_whisper import WhisperModel
from piper import PiperVoice, SynthesisConfig

import overlay_ipc

# ----------------------------------------------------------------------
# Config — tweak these for your machine
# ----------------------------------------------------------------------

SAMPLE_RATE = 16000
FRAME_SAMPLES = 1280  # 80 ms — openWakeWord requires multiples of this

# Wake word. Defaults to openWakeWord's pretrained "hey jarvis" model.
# To use your own phrase (e.g. "Hey Nyx"), train a custom model — see
# wake_word_training/README.md — then point these three at it:
#   WAKE_WORD_MODEL_PATH -> path to your trained .onnx file
#   WAKE_WORD_KEY         -> the key it reports (its filename, no extension)
#   WAKE_PHRASE           -> just cosmetic, printed in the "Ready" messages
WAKE_WORD_MODEL_PATH = "hey_nyx.onnx"
WAKE_WORD_KEY = "hey_nyx"
WAKE_PHRASE = "Hey Nyx"
WAKE_THRESHOLD = 0.01

# Ollama — check `ollama list` and match the exact tag you pulled
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "gemma2:2b-instruct-q4_K_M"

# How many tokens Ollama is allowed to generate per reply. Keeps latency
# down for a voice assistant — long replies are slow to generate AND
# awkward to sit through as speech. Raise if replies feel cut off.
OLLAMA_MAX_TOKENS = 120

# faster-whisper — TUF A15 has an NVIDIA GPU, so use it. Falls back to
# WHISPER_DEVICE="cpu", WHISPER_COMPUTE="int8" if CUDA isn't cooperating.
WHISPER_MODEL_SIZE = "small"
WHISPER_DEVICE = "cuda"
WHISPER_COMPUTE = "float16"

# Piper voice — download a .onnx + .onnx.json pair into voices/ (see README)
PIPER_MODEL_PATH = "voices/en_US-lessac-medium.onnx"
PIPER_SYN_CONFIG = SynthesisConfig(length_scale=1.0)

# Audio input device — "default" wasn't reliably passing mic input through
# on this pipewire setup, so pin it explicitly. Re-check with
# `python -c "import sounddevice as sd; print(sd.query_devices())"` if audio
# breaks after a reboot or device change.
AUDIO_INPUT_DEVICE = "pipewire"
MAX_RECORD_SECONDS = 10

# VAD (webrtcvad) replaces the old fixed RMS threshold for silence
# detection. Aggressiveness 0-3; higher = more aggressively classifies
# audio as non-speech (fewer false positives from background noise, but
# can clip quiet speech). 2 is a reasonable middle ground.
VAD_AGGRESSIVENESS = 2
VAD_SUBFRAME_SAMPLES = 320  # 20 ms @ 16kHz — the frame size webrtcvad wants
SILENCE_HANG_FRAMES = 25    # ~2s of quiet (25 * 80ms) before we stop recording

# mpv IPC socket, used for stop/pause/resume/skip/volume tool commands.
MPV_IPC_SOCKET = "/tmp/nyx-mpv.sock"

# Where conversation history is persisted between runs.
HISTORY_FILE = Path.home() / ".local" / "share" / "nyx-assistant" / "history.json"

SYSTEM_PROMPT = (
    "You are Nyx, a terse, dry-witted local voice assistant running "
    "entirely on the user's own machine. Keep replies short (1-3 sentences) "
    "unless asked for detail, since they'll be read aloud by a TTS engine. "
    "IMPORTANT: you never generate replies about playing, stopping, "
    "pausing, resuming, or skipping music, changing its volume, opening/"
    "launching apps, websites, or folders, closing apps, changing the "
    "wallpaper, searching the web, telling the time or date, turning "
    "wifi, bluetooth, or night light on or off, or changing the system "
    "volume, muting/unmuting, or changing screen brightness, playing "
    "music on Spotify, or reading/summarizing a browser page — "
    "separate keyword-matching systems handle those commands directly "
    "and you are never even shown those turns unless they fail to match, "
    "in which case just respond as normal conversation and do NOT claim "
    "you played, stopped, opened, launched, closed, changed the "
    "wallpaper, searched for anything, told the time, toggled wifi, "
    "bluetooth, or night light, changed the volume, muted/unmuted, "
    "changed the brightness, played something on Spotify, or read/"
    "summarized a browser page. If "
    "earlier turns in this chat show messages like 'Playing X.', "
    "'Stopped.', 'Opening X.', or 'Searching for X.', those came from "
    "those separate systems, not from you — do not imitate that pattern "
    "or narrate those actions yourself. You have no tools of your own "
    "beyond those whitelisted systems: don't claim you can set "
    "reminders, control smart home devices, open anything not already "
    "in that whitelist, search the web yourself and report results, or "
    "run other commands."
)

MAX_HISTORY_TURNS = 20  # messages kept, not counting the system prompt

# Hard backstop matching the system prompt's "1-3 sentences" instruction.
# Even with a well-formed prompt, a small/abliterated model can still go
# off-script — this stops us from printing/speaking a runaway reply.
MAX_REPLY_SENTENCES = 3


# ----------------------------------------------------------------------
# Persistent memory
# ----------------------------------------------------------------------

def load_history() -> list[dict]:
    try:
        data = json.loads(HISTORY_FILE.read_text())
        if isinstance(data, list) and data:
            data[0] = {"role": "system", "content": SYSTEM_PROMPT}  # keep prompt current
            return data
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        pass
    return [{"role": "system", "content": SYSTEM_PROMPT}]


def save_history(history: list[dict]) -> None:
    try:
        HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
        HISTORY_FILE.write_text(json.dumps(history))
    except OSError as e:
        print(f"Warning: couldn't save history ({e})", file=sys.stderr)


def trim_history(history: list[dict]) -> None:
    if len(history) > MAX_HISTORY_TURNS + 1:
        history[:] = [history[0]] + history[-MAX_HISTORY_TURNS:]


def llm_messages(history: list[dict]) -> list[dict]:
    """The message list actually sent to Ollama. Tool-handled turns keep
    their place in the alternating user/assistant structure — dropping
    them entirely (the old behavior) left back-to-back user turns
    whenever two tool commands happened in a row, and small models tend
    to "fix" a broken alternating pattern by hallucinating the missing
    turns themselves, which is what was producing whole fake exchanges.
    The actual spoken reply ("Playing X.", "Stopped.") is still swapped
    out for a neutral placeholder, so the model isn't shown that pattern
    to imitate — it just sees that *something* was handled."""
    out = []
    for m in history:
        if m.get("tool"):
            out.append({"role": "assistant", "content": "[handled by a separate system]"})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


# ----------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------

def play_earcon() -> None:
    """Short two-note chime played the instant the wake word fires, so
    there's audible confirmation Nyx is listening before you start
    talking — no need to guess whether it heard you. Synthesized on the
    fly (no sound file to manage); ~150ms total, quiet by design so it
    doesn't feel like a notification blast."""
    notes = [(700, 0.06), (1050, 0.08)]  # (Hz, seconds) — short rising "doo-dit"
    chunks = []
    for freq, duration in notes:
        t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
        tone = 0.15 * np.sin(2 * np.pi * freq * t).astype(np.float32)
        fade = min(int(0.01 * SAMPLE_RATE), len(tone) // 2)
        if fade > 0:
            tone[:fade] *= np.linspace(0, 1, fade, dtype=np.float32)
            tone[-fade:] *= np.linspace(1, 0, fade, dtype=np.float32)
        chunks.append(tone)
        chunks.append(np.zeros(int(0.02 * SAMPLE_RATE), dtype=np.float32))  # tiny gap
    sd.play(np.concatenate(chunks), SAMPLE_RATE)
    sd.wait()


def drain_queue(q: queue.Queue) -> None:
    """Discard any buffered mic audio (e.g. audio queued before we start
    actively listening/recording)."""
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def is_speech(pcm: np.ndarray, vad: webrtcvad.Vad) -> bool:
    """Majority-vote speech detection over 20ms sub-frames of an 80ms frame."""
    votes = 0
    total = 0
    for i in range(0, len(pcm) - VAD_SUBFRAME_SAMPLES + 1, VAD_SUBFRAME_SAMPLES):
        sub = pcm[i:i + VAD_SUBFRAME_SAMPLES].tobytes()
        total += 1
        if vad.is_speech(sub, SAMPLE_RATE):
            votes += 1
    return total > 0 and votes >= (total // 2 + 1)


def _level_from_pcm(pcm: np.ndarray, gain: float = 6.0) -> float:
    """Rough 0-1 amplitude for the overlay widget, from a raw int16 frame."""
    rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)) / 32768.0)
    return min(1.0, rms * gain)


def record_command(
    audio_q: queue.Queue,
    vad: webrtcvad.Vad,
    initial_timeout: float,
    max_total: float,
) -> np.ndarray:
    """Record from audio_q until VAD detects silence following speech, or
    max_total elapses. If no speech starts within initial_timeout, give up
    early and return empty (used to keep the follow-up window short)."""
    frames = []
    silence_run = 0
    heard_speech = False
    start = time.time()

    while time.time() - start < max_total:
        if not heard_speech and time.time() - start > initial_timeout:
            break
        try:
            raw = audio_q.get(timeout=0.2)
        except queue.Empty:
            continue
        pcm = np.frombuffer(raw, dtype=np.int16)
        frames.append(pcm)
        overlay_ipc.send_state("listening", level=_level_from_pcm(pcm))

        if is_speech(pcm, vad):
            heard_speech = True
            silence_run = 0
        elif heard_speech:
            silence_run += 1
            if silence_run > SILENCE_HANG_FRAMES:
                break

    if not heard_speech:
        return np.array([], dtype=np.float32)

    return np.concatenate(frames).astype(np.float32) / 32768.0


def transcribe(whisper: WhisperModel, audio: np.ndarray) -> str:
    # beam_size=1/best_of=1 (greedy) is fastest but mishears things easily
    # ("Blinding Lights" -> "Blending Lights for weekend" territory). A
    # bigger beam costs a bit of latency but is meaningfully more accurate,
    # and your 3050 has room for it alongside Whisper-small.
    #
    # Whisper also has a known failure mode: fed a short noise/silence
    # blip, it will happily invent text ("You", "Thank you.") instead of
    # saying "nothing here." vad_filter strips non-speech before decoding,
    # and the no_speech_prob/avg_logprob check below throws out whatever
    # low-confidence segments still sneak through.
    if audio.size < int(0.4 * SAMPLE_RATE):
        return ""  # too short to plausibly be real speech
    segments, _ = whisper.transcribe(
        audio,
        language="en",
        beam_size=5,
        best_of=5,
        vad_filter=True,
        vad_parameters={"min_silence_duration_ms": 300},
    )
    good = [s.text for s in segments if s.no_speech_prob < 0.6 and s.avg_logprob > -1.0]
    return " ".join(good).strip()


def speak_interruptible(
    voice: PiperVoice,
    text: str,
    audio_q: queue.Queue,
    ww_model: WakeWordModel,
) -> bool:
    """Speak text through Piper. If the wake word is heard while Nyx is
    still talking, stop playback immediately (barge-in) and return True."""
    if not text.strip():
        return False
    chunks = list(voice.synthesize(text, PIPER_SYN_CONFIG))
    if not chunks:
        return False
    audio = np.concatenate([c.audio_float_array for c in chunks])
    sample_rate = chunks[0].sample_rate

    drain_queue(audio_q)  # ignore anything queued before we started talking
    sd.play(audio, sample_rate)
    playback_start = time.time()

    interrupted = False
    stream = sd.get_stream()
    while stream is not None and stream.active:
        elapsed = time.time() - playback_start
        pos = int(elapsed * sample_rate)
        window = audio[max(0, pos - 800):pos]  # ~50ms window for the level
        level = min(1.0, float(np.sqrt(np.mean(window ** 2))) * 3.0) if window.size else 0.0
        overlay_ipc.send_state("speaking", level=level)

        try:
            raw = audio_q.get(timeout=0.1)
        except queue.Empty:
            continue
        frame = np.frombuffer(raw, dtype=np.int16)
        if len(frame) != FRAME_SAMPLES:
            continue
        prediction = ww_model.predict(frame)
        score = prediction.get(WAKE_WORD_KEY, 0.0)
        if score > WAKE_THRESHOLD:
            sd.stop()
            ww_model.reset()
            interrupted = True
            overlay_ipc.send_state("listening")
            break

    if not interrupted:
        sd.wait()
    drain_queue(audio_q)
    return interrupted


# ----------------------------------------------------------------------
# Ollama — streaming
# ----------------------------------------------------------------------

def ask_ollama_stream(messages: list[dict]):
    """Yields text deltas from Ollama's streaming chat endpoint."""
    with requests.post(
        OLLAMA_URL,
        json={
            "model": OLLAMA_MODEL,
            "messages": messages,
            "stream": True,
            "keep_alive": "10m",
            "options": {
                "num_predict": OLLAMA_MAX_TOKENS,
                "temperature": 0.3,
                "repeat_penalty": 1.3,
                # Belt-and-suspenders: if the model ever starts a new
                # speaker turn instead of just answering, cut it off there.
                "stop": ["\nYou:", "\nUser:", "\nNyx:"],
            },
        },
        timeout=120,
        stream=True,
    ) as resp:
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            delta = chunk.get("message", {}).get("content", "")
            if delta:
                yield delta
            if chunk.get("done"):
                break


SENTENCE_END_RE = re.compile(r"[.!?](\s|$)")


def stream_sentences(token_iter):
    """Buffers streamed text deltas and yields complete sentences as soon
    as they're available, so Piper can start speaking early."""
    buf = ""
    for delta in token_iter:
        buf += delta
        while True:
            m = SENTENCE_END_RE.search(buf)
            if not m:
                break
            sentence, buf = buf[:m.end()].strip(), buf[m.end():]
            if sentence:
                yield sentence
    if buf.strip():
        yield buf.strip()


# ----------------------------------------------------------------------
# Flavor lines — purely cosmetic. Each real tool action still runs
# exactly the same regardless of which line gets picked; this only
# changes what Nyx *says* about it, chosen at random from a small pool
# per action so she doesn't repeat herself every time. Add more lines to
# any list below to taste — nothing else needs to change. `{}`-style
# placeholders are filled in by quip()'s kwargs (e.g. label=, name=).
# ----------------------------------------------------------------------

QUIPS: dict[str, list[str]] = {
    "open": [
        "Domain Expansion: {label}.",
        "Unlimited {label} Works.",
        "Summoning {label} from the shadow realm.",
        "Opening {label}. No cursed energy wasted.",
    ],
    "close": [
        "Domain collapsed. {label}, banished.",
        "Closing {label}. Exorcism complete.",
        "{label} sent back to the shadow realm.",
    ],
    "wallpaper_random": [
        "Infinite Void: fetching a random wallpaper.",
        "Peering into the Infinite Void for something new.",
        "Domain Expansion: Random Wallpaper Realm.",
    ],
    "wallpaper_set": [
        "Domain Expansion: {name}.",
        "Reality reshaped — wallpaper now {name}.",
        "Cursed technique applied: {name}.",
    ],
    "play_music": [
        "Domain Expansion: {query}, now playing.",
        "Summoning {query} from the shadow realm.",
        "Cursed technique activated: {query}.",
    ],
    "spotify_play": [
        "Domain Expansion: {query}, streaming from Spotify.",
        "Cursed technique, Spotify style: {query}.",
        "Summoning {query} from the Spotify realm.",
    ],
    "stop_music": [
        "Domain collapsed. Silence restored.",
        "Cursed energy severed. Stopped.",
    ],
    "search": [
        "Domain Expansion: Search Realm — looking up {query}.",
        "Consulting the Infinite Void about {query}.",
    ],
}


def quip(key: str, default: str, **kwargs) -> str:
    """Pick a random flavor line for `key` and fill in kwargs. Falls back
    to `default` (the old plain reply) if `key` isn't registered, so a
    typo here never breaks a tool reply."""
    options = QUIPS.get(key)
    if not options:
        return default
    return random.choice(options).format(**kwargs)


# ----------------------------------------------------------------------
# Tools — small, explicit, keyword-matched actions. Deliberately NOT a
# "let the LLM run arbitrary shell commands" setup: each tool below does
# exactly one known-safe thing. Add more the same way as you need them.
#
# Caveat: these are simple keyword matches, so phrases like "don't stop
# believing" or "what's next on my calendar" could false-trigger the
# media tools instead of going to the LLM. Fine for a personal project;
# tighten the regexes if it becomes annoying.
# ----------------------------------------------------------------------

PLAY_MUSIC_RE = re.compile(
    r"\bplay\s+(?:me\s+|some\s+)?(.+?)(?:\s+on\s+youtube)?[.!?]*$", re.IGNORECASE
)
STOP_RE = re.compile(r"\b(stop|shut up|quiet)\b", re.IGNORECASE)
PAUSE_RE = re.compile(r"\bpause\b", re.IGNORECASE)
RESUME_RE = re.compile(r"\b(resume|unpause|continue)\b", re.IGNORECASE)
SKIP_RE = re.compile(r"\b(skip|next)\b", re.IGNORECASE)


# ----------------------------------------------------------------------
# Hyprland (hyprctl) helpers — used for workspace/window awareness,
# closing apps by voice, and launching new apps onto a fresh workspace.
# ----------------------------------------------------------------------

def hyprctl(*args: str) -> None:
    """Fire-and-forget a hyprctl dispatch command."""
    try:
        subprocess.run(
            ["hyprctl", *args],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
        )
    except (subprocess.SubprocessError, FileNotFoundError):
        pass


def hyprctl_clients() -> list[dict]:
    """All open windows across ALL workspaces (not just the focused one),
    via `hyprctl clients -j`. Returns [] if hyprctl isn't reachable —
    callers should treat that as "no match" and fall through gracefully,
    not raise."""
    try:
        out = subprocess.run(
            ["hyprctl", "-j", "clients"],
            capture_output=True, text=True, timeout=2,
        )
        return json.loads(out.stdout)
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return []



def mpv_command(*args: str) -> bool:
    """Send a command to the running mpv instance over its IPC socket.
    Returns True if it was sent (i.e. mpv is running), False otherwise."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            s.connect(MPV_IPC_SOCKET)
            s.sendall(json.dumps({"command": list(args)}).encode() + b"\n")
        print(f"[mpv] sent {args} OK")
        return True
    except OSError as e:
        print(f"[mpv] {args} failed: {e}", file=sys.stderr)
        return False


_mpv_process: subprocess.Popen | None = None


def play_music(query: str) -> None:
    global _mpv_process
    # Only one mpv instance should ever be alive at a time. Without this,
    # saying "play X" then later "play Y" leaves the first mpv running as
    # an orphan — the new one claims the IPC socket, so "stop" can only
    # ever reach whichever instance is newest, and the old one keeps
    # playing in the background with no way to control it.
    if _mpv_process is not None and _mpv_process.poll() is None:
        _mpv_process.terminate()
        try:
            _mpv_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _mpv_process.kill()

    if os.path.exists(MPV_IPC_SOCKET):
        os.remove(MPV_IPC_SOCKET)

    _mpv_process = subprocess.Popen(
        [
            "mpv",
            "--no-video",
            f"--input-ipc-server={MPV_IPC_SOCKET}",
            "--script-opts=ytdl_hook-ytdl_path=yt-dlp",
            f"ytdl://ytsearch:{query}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_music() -> bool:
    """Stop playback AND make sure the mpv process is actually gone — not
    just told to stop — so nothing can keep playing in the background
    regardless of the user's own mpv.conf (e.g. an idle=yes setting would
    otherwise leave the process alive and silent instead of exiting).
    Returns True if there was anything to stop."""
    global _mpv_process
    sent = mpv_command("stop")
    if _mpv_process is not None and _mpv_process.poll() is None:
        _mpv_process.terminate()
        try:
            _mpv_process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            _mpv_process.kill()
        sent = True
    return sent


# ----------------------------------------------------------------------
# System launch tools — whitelisted apps, websites, and folders only.
# Same guiding rule as the media tools above: Nyx can never run an
# arbitrary command. A mis-heard Whisper transcription or a hallucinated
# tool call can only ever match something already in OPEN_WHITELIST
# below — anything else silently falls through to a normal LLM reply,
# nothing runs.
#
# EDIT THIS to match what's actually installed on your machine. The
# defaults below assume Firefox, kitty (end-4/dots-hyprland's default
# terminal), and Dolphin — check with e.g. `which kitty` and swap the
# argv list if yours differ. Add more entries the same way; no other
# code needs to change.
# ----------------------------------------------------------------------

# alias phrase (lowercase, already stripped of filler words) -> (kind, target, spoken label)
#   "app"  -> target is an argv list passed straight to subprocess.Popen
#   "url"  -> target is a URL, opened via xdg-open (your actual default
#             browser, whatever that's set to)
#   "path" -> target is a filesystem path, also opened via xdg-open,
#             which hands it to your default file manager
OPEN_WHITELIST: dict[str, tuple[str, list[str] | str, str]] = {
    # --- apps ---
    "firefox": ("app", ["firefox"], "Firefox"),
    "fire fox": ("app", ["firefox"], "Firefox"),
    "fire folks": ("app", ["firefox"], "Firefox"),  # common Whisper mis-hearing
    "far fox": ("app", ["firefox"], "Firefox"),      # common Whisper mis-hearing
    "browser": ("app", ["firefox"], "Firefox"),
    "web browser": ("app", ["firefox"], "Firefox"),
    "terminal": ("app", ["kitty"], "the terminal"),
    "console": ("app", ["kitty"], "the terminal"),
    "discord": ("app", ["discord"], "Discord"),
    "spotify": ("app", ["spotify"], "Spotify"),
    "file manager": ("app", ["dolphin"], "the file manager"),
    "files": ("app", ["dolphin"], "the file manager"),
    "dolphin": ("app", ["dolphin"], "the file manager"),
    "code": ("app", ["code"], "VS Code"),
    "vscode": ("app", ["code"], "VS Code"),
    "vs code": ("app", ["code"], "VS Code"),
    "text editor": ("app", ["code"], "VS Code"),
    # --- websites ---
    "youtube": ("url", "https://youtube.com", "YouTube"),
    "github": ("url", "https://github.com", "GitHub"),
    "reddit": ("url", "https://reddit.com", "Reddit"),
    "gmail": ("url", "https://mail.google.com", "Gmail"),
    "email": ("url", "https://mail.google.com", "Gmail"),
    # --- folders ---
    "downloads": ("path", "~/Downloads", "your Downloads folder"),
    "documents": ("path", "~/Documents", "your Documents folder"),
    "pictures": ("path", "~/Pictures", "your Pictures folder"),
    "home": ("path", "~", "your home folder"),
}

OPEN_RE = re.compile(r"\b(?:open|launch|start)\s+(.+?)[.!?]*$", re.IGNORECASE)

# Small talk fillers stripped from both ends of the captured phrase before
# whitelist lookup, e.g. "open up my downloads folder please" -> "downloads".
# Deliberately word-level and only trims leading/trailing matches, so it
# can't accidentally mangle a real multi-word key like "file manager".
_OPEN_FILLER_WORDS = {
    "the", "a", "an", "my", "up", "please", "now", "thanks", "thank",
    "you", "for", "me", "app", "application", "website", "site",
    "folder", "directory", "dir",
}


def _clean_open_phrase(phrase: str) -> str:
    words = phrase.split()
    while words and words[0] in _OPEN_FILLER_WORDS:
        words.pop(0)
    while words and words[-1] in _OPEN_FILLER_WORDS:
        words.pop()
    return " ".join(words)


def try_handle_open(text: str) -> str | None:
    """Matches 'open/launch/start <whitelisted thing>'. Only ever acts on
    an exact match in OPEN_WHITELIST — an unrecognized phrase returns
    None (falls through to a normal LLM reply) rather than guessing at a
    command or running anything free-form."""
    match = OPEN_RE.search(text)
    if not match:
        return None

    phrase = _clean_open_phrase(match.group(1).strip().lower())
    entry = OPEN_WHITELIST.get(phrase)
    if entry is None:
        return None

    kind, target, label = entry
    try:
        if kind == "app":
            # Jump to the first empty workspace on the current monitor
            # *before* spawning — new windows land on whatever workspace
            # is focused, so this is what gives each opened app its own
            # workspace instead of piling onto whatever you're already on.
            hyprctl("dispatch", "workspace", "emptym")
            subprocess.Popen(target, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:  # "url" or "path" — both handed to the desktop's default handler
            hyprctl("dispatch", "workspace", "emptym")
            dest = os.path.expanduser(target) if kind == "path" else target
            subprocess.Popen(["xdg-open", dest], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return f"Couldn't find {label} — check it's installed and the binary name matches."

    return quip("open", f"Opening {label}.", label=label)


CLOSE_RE = re.compile(r"\bclose\s+(.+?)[.!?]*$", re.IGNORECASE)


def try_handle_close(text: str) -> str | None:
    """Matches 'close <app>'. Looks at ALL currently open windows across
    every workspace (hyprctl clients, not just the focused workspace) and
    closes the first one whose window class or title matches. Same
    closed-whitelist spirit as the rest of this file: it only ever acts
    on a real window that's actually open right now — an unrecognized or
    no-longer-open app name returns None and falls through to the LLM,
    it never guesses or runs anything else."""
    match = CLOSE_RE.search(text)
    if not match:
        return None
    phrase = _clean_open_phrase(match.group(1).strip().lower())
    if not phrase:
        return None

    # If the phrase is a known OPEN_WHITELIST alias, also search for the
    # underlying binary name (e.g. "browser" -> also try "firefox") so
    # "close the browser" finds a window launched as "firefox".
    search_terms = [phrase]
    label = phrase
    entry = OPEN_WHITELIST.get(phrase)
    if entry and entry[0] == "app":
        search_terms.append(entry[1][0].lower())
        label = entry[2]

    for client in hyprctl_clients():
        cls = str(client.get("class", "")).lower()
        title = str(client.get("title", "")).lower()
        address = client.get("address")
        if not address:
            continue
        if any(term and (term in cls or term in title) for term in search_terms):
            hyprctl("dispatch", "closewindow", f"address:{address}")
            return quip("close", f"Closing {label}.", label=label)

    return None  # nothing open matches — let the LLM respond normally


# ----------------------------------------------------------------------
# Wallpaper — end-4/dots-hyprland (illogical-impulse) specific. Random
# wallpaper goes through Hyprland's own global-shortcut IPC target,
# same as the "Random: Konachan"/"Random: osu! seasonal" buttons in
# Settings. Picking a specific wallpaper by name shells out to the same
# switchwall.sh script the "Choose file" button uses, which also
# triggers the Material color regeneration for theming.
#
# EDIT THESE two paths for your setup:
#   WALLPAPER_DIR       -> folder of wallpaper image files to match names
#                           against ("change wallpaper to konachan girl")
#                           — also where random_konachan_wall.sh saves to
#   SWITCHWALL_SCRIPT    -> should already be correct on a stock ii install
#   RANDOM_WALLPAPER_SCRIPT -> confirmed via `grep -rn konachan
#                           ~/.config/quickshell/ii/` — this is the exact
#                           script the "Random: Konachan" Settings button
#                           runs (fetches a random SFW anime wallpaper
#                           from Konachan into WALLPAPER_DIR and applies it)
#
# NOTE: switchwall.sh's normal flow opens a yad file-picker UI rather
# than taking a path argument out of the box on stock ii installs. Test
# manually first: `~/.config/quickshell/ii/scripts/colors/switchwall.sh
# /path/to/some/wallpaper.jpg` in a terminal. If it pops the picker
# instead of applying that file directly, you'll want to either patch
# switchwall.sh to accept $1 as imgpath (skip yad when an arg is given),
# or point SWITCHWALL_SCRIPT at your own small wrapper that does that.
# ----------------------------------------------------------------------

WALLPAPER_DIR = Path.home() / "Pictures" / "Wallpapers"
SWITCHWALL_SCRIPT = Path.home() / ".config" / "quickshell" / "ii" / "scripts" / "colors" / "switchwall.sh"
RANDOM_WALLPAPER_SCRIPT = (
    Path.home() / ".config" / "quickshell" / "ii" / "scripts" / "colors" / "random" / "random_konachan_wall.sh"
)

RANDOM_WALLPAPER_RE = re.compile(r"\b(?:random|new)\s+wallpaper\b", re.IGNORECASE)
WALLPAPER_RE = re.compile(
    r"\b(?:change|set)\s+(?:the\s+)?wallpaper(?:\s+to\s+(.+?))?[.!?]*$", re.IGNORECASE
)


def try_handle_wallpaper(text: str) -> str | None:
    if RANDOM_WALLPAPER_RE.search(text):
        try:
            subprocess.Popen(
                [str(RANDOM_WALLPAPER_SCRIPT)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return "Couldn't run the random wallpaper script — check RANDOM_WALLPAPER_SCRIPT's path."
        return quip("wallpaper_random", "Fetching a random wallpaper.")

    match = WALLPAPER_RE.search(text)
    if not match:
        return None
    name = (match.group(1) or "").strip().lower()
    if not name:
        # "change the wallpaper" with no target named — treat as random.
        try:
            subprocess.Popen(
                [str(RANDOM_WALLPAPER_SCRIPT)],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return "Couldn't run the random wallpaper script — check RANDOM_WALLPAPER_SCRIPT's path."
        return quip("wallpaper_random", "Fetching a random wallpaper.")

    if not WALLPAPER_DIR.is_dir():
        return "Couldn't find your wallpaper folder — check WALLPAPER_DIR in main.py."

    hit = next(
        (p for p in WALLPAPER_DIR.iterdir() if p.is_file() and name in p.stem.lower()),
        None,
    )
    if hit is None:
        return f"Couldn't find a wallpaper matching {name}."

    try:
        subprocess.Popen(
            [str(SWITCHWALL_SCRIPT), str(hit)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return "Couldn't run the wallpaper switch script — check SWITCHWALL_SCRIPT's path."
    return quip("wallpaper_set", f"Changing wallpaper to {hit.stem}.", name=hit.stem)


# ----------------------------------------------------------------------
# Spotify — "open spotify and play X" / "play X on spotify". Uses the
# ALREADY-RUNNING desktop app, not Spotify's remote-playback Web API —
# no browser login flow, no Premium requirement, no polling for a
# "device". Two pieces:
#   1. Track lookup: Spotify's /v1/search endpoint, using a
#      client-credentials token (app-only, no user login — just proves
#      "some registered app is asking"). Free at
#      https://developer.spotify.com/dashboard: create an app, no
#      redirect URI needed since there's no user login step at all.
#   2. Playback: the desktop app exposes an MPRIS D-Bus interface once
#      it's running (org.mpris.MediaPlayer2.spotify) — its OpenUri
#      method takes a spotify:track:... URI and just plays it, same as
#      if you'd clicked play yourself. `dbus-send` (part of the base
#      dbus package, already on virtually every Linux install) is used
#      rather than a Python D-Bus library, to avoid adding a dependency.
#
# Nothing here ever runs arbitrary code from the query — it's only ever
# used as a search string against Spotify's API, same spirit as the
# other free-form tools (play_music, search).
# ----------------------------------------------------------------------

SPOTIFY_CONFIG_FILE = Path.home() / ".config" / "nyx-assistant" / "spotify.json"
SPOTIFY_SEARCH_URL = "https://api.spotify.com/v1/search"
SPOTIFY_TOKEN_URL = "https://accounts.spotify.com/api/token"
SPOTIFY_DBUS_DEST = "org.mpris.MediaPlayer2.spotify"

# In-memory access-token cache — client-credentials tokens are app-wide
# (not tied to any user), so one token comfortably covers every search
# until it expires (usually 1hr).
_spotify_token_cache: dict = {"access_token": None, "expires_at": 0.0}


def _load_spotify_config() -> dict | None:
    """Expects {"client_id": "...", "client_secret": "..."} — see
    SETUP_NEW_FEATURES.md for how to get those."""
    try:
        return json.loads(SPOTIFY_CONFIG_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _spotify_access_token(cfg: dict) -> str | None:
    if _spotify_token_cache["access_token"] and time.time() < _spotify_token_cache["expires_at"] - 60:
        return _spotify_token_cache["access_token"]

    try:
        resp = requests.post(
            SPOTIFY_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(cfg["client_id"], cfg["client_secret"]),
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
    except (requests.exceptions.RequestException, KeyError):
        return None

    _spotify_token_cache["access_token"] = data.get("access_token")
    _spotify_token_cache["expires_at"] = time.time() + data.get("expires_in", 3600)
    return _spotify_token_cache["access_token"]


def _spotify_search_track_uri(query: str, token: str) -> str | None:
    try:
        resp = requests.get(
            SPOTIFY_SEARCH_URL,
            headers={"Authorization": f"Bearer {token}"},
            params={"q": query, "type": "track", "limit": 1},
            timeout=10,
        )
        resp.raise_for_status()
        items = resp.json().get("tracks", {}).get("items", [])
    except requests.exceptions.RequestException:
        return None
    return items[0]["uri"] if items else None


def _spotify_dbus_ready() -> bool:
    """True once the desktop app's MPRIS D-Bus interface answers — a
    freshly launched app takes a couple seconds to register it."""
    try:
        out = subprocess.run(
            ["dbus-send", "--session", "--print-reply",
             f"--dest={SPOTIFY_DBUS_DEST}",
             "/org/mpris/MediaPlayer2", "org.freedesktop.DBus.Peer.Ping"],
            capture_output=True, timeout=2,
        )
        return out.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def _ensure_spotify_app(timeout: float = 8.0) -> bool:
    """Launches the desktop app if it isn't already running, then waits
    for its D-Bus interface to come up. Returns False if it never does
    within `timeout`."""
    if not _process_running("spotify"):
        try:
            subprocess.Popen(["spotify"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except FileNotFoundError:
            return False

    deadline = time.time() + timeout
    while time.time() < deadline:
        if _spotify_dbus_ready():
            return True
        time.sleep(0.5)
    return False


def _spotify_dbus_open_uri(track_uri: str) -> bool:
    try:
        out = subprocess.run(
            ["dbus-send", "--session", "--type=method_call",
             f"--dest={SPOTIFY_DBUS_DEST}", "/org/mpris/MediaPlayer2",
             "org.mpris.MediaPlayer2.Player.OpenUri", f"string:{track_uri}"],
            capture_output=True, timeout=5,
        )
        return out.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def spotify_search_and_play(query: str) -> tuple[bool, str]:
    """Searches Spotify for `query` and tells the already-running (or
    freshly launched) desktop app to play it via D-Bus. Returns
    (success, message) — message is the error to speak on failure, or
    ignored on success (caller supplies its own quip line)."""
    cfg = _load_spotify_config()
    if cfg is None:
        return False, "Spotify search isn't set up yet — check SETUP_NEW_FEATURES.md."

    token = _spotify_access_token(cfg)
    if token is None:
        return False, "Couldn't authenticate with Spotify — check your saved credentials."

    track_uri = _spotify_search_track_uri(query, token)
    if track_uri is None:
        return False, f"Couldn't find {query} on Spotify."

    if not _ensure_spotify_app():
        return False, "Couldn't reach the Spotify app — is it installed?"

    if not _spotify_dbus_open_uri(track_uri):
        return False, "Spotify wouldn't take that track — check dbus-send is installed."

    return True, "ok"


SPOTIFY_OPEN_PLAY_RE = re.compile(
    r"\bopen\s+spotify\s+and\s+play\s+(.+?)[.!?]*$", re.IGNORECASE
)
SPOTIFY_PLAY_ON_RE = re.compile(
    r"\bplay\s+(.+?)\s+on\s+spotify[.!?]*$", re.IGNORECASE
)


def try_handle_spotify(text: str) -> str | None:
    """Must be checked before the generic PLAY_MUSIC_RE tool below —
    otherwise "open spotify and play X" gets caught by the plain
    play-via-mpv handler instead, since its regex matches "play ..."
    anywhere in the utterance."""
    match = SPOTIFY_OPEN_PLAY_RE.search(text) or SPOTIFY_PLAY_ON_RE.search(text)
    if not match:
        return None
    query = match.group(1).strip()
    if not query:
        return None

    ok, msg = spotify_search_and_play(query)
    if not ok:
        return msg
    return quip("spotify_play", f"Playing {query} on Spotify.", query=query)


# "Search for X" / "search X" — unlike OPEN_WHITELIST above, this one
# takes free-form text (like play_music does for song queries), so it
# isn't a closed whitelist of phrases. It's still safe in the same way
# play_music is: the captured text is never executed as a command, only
# URL-encoded into a fixed google.com search URL and handed to
# xdg-open — so at worst a bad transcription searches for the wrong
# thing, it can't run anything.
SEARCH_RE = re.compile(r"\bsearch(?:\s+for)?\s+(.+?)[.!?]*$", re.IGNORECASE)


def try_handle_search(text: str) -> str | None:
    match = SEARCH_RE.search(text)
    if not match:
        return None
    query = match.group(1).strip()
    if not query:
        return None

    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    try:
        subprocess.Popen(["xdg-open", url], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        return "Couldn't open a browser to search — check xdg-open is set up."
    return quip("search", f"Searching for {query}.", query=query)


# ----------------------------------------------------------------------
# Date/time — answered straight from the system clock, no LLM or
# subprocess involved. %-I / %-d (no leading zero) are glibc/Linux-
# specific strftime extensions; fine here since this is a Linux-only
# assistant, but note they won't work as-is on macOS/BSD's libc.
# ----------------------------------------------------------------------

TIME_RE = re.compile(
    r"\bwhat(?:'s|s|\s+is)\s+(?:the\s+)?(?:current\s+)?time\b|\bwhat\s+time\s+is\s+it\b",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\bwhat(?:'s|s|\s+is)\s+(?:the\s+)?(?:current\s+)?date\b"
    r"|\bwhat\s+day\s+is\s+it\b"
    r"|\bwhat(?:'s|s|\s+is)\s+today(?:'s\s+date)?\b",
    re.IGNORECASE,
)


def try_handle_datetime(text: str) -> str | None:
    if TIME_RE.search(text):
        return f"It's {datetime.now().strftime('%-I:%M %p')}."
    if DATE_RE.search(text):
        return f"Today is {datetime.now().strftime('%A, %B %-d')}."
    return None


# ----------------------------------------------------------------------
# Browser info (Firefox) — two tiers:
#
#   1. "what tab/page am I on" — free, zero-setup. Hyprland already
#      reports the focused window's title via `hyprctl activewindow`,
#      and in Firefox that title *is* the current tab's title. Good
#      enough for a quick "what am I looking at".
#
#   2. "summarize this page" / "read this page" — needs the companion
#      Firefox extension (nyx_browser_extension/) installed, plus its
#      native messaging host running. The extension grabs the active
#      tab's title/URL/visible text and relays it to Nyx over a Unix
#      socket (same IPC-over-socket pattern as overlay_ipc.py), and Nyx
#      feeds that through a one-off Ollama call for a short summary —
#      this call is deliberately NOT added to conversation history, so
#      it can't bloat context with page text.
#
# Neither tier ever sends anything back to the page or executes
# anything the page contains — it's read-only, title/URL/visible-text
# extraction only.
# ----------------------------------------------------------------------

BROWSER_WINDOW_CLASSES = {"firefox", "firefox-esr", "librewolf", "floorp"}
BROWSER_IPC_SOCKET = "/tmp/nyx-browser.sock"

BROWSER_TAB_RE = re.compile(
    r"\bwhat\s+(?:tab|page)\s+am\s+i\s+on\b"
    r"|\bwhat(?:'s|\s+is)\s+(?:on\s+my\s+screen|this\s+(?:tab|page))\b"
    r"|\bwhat\s+am\s+i\s+looking\s+at\b",
    re.IGNORECASE,
)
BROWSER_SUMMARY_RE = re.compile(
    r"\b(?:summarize|summarise)\s+this\s+page\b"
    r"|\bread\s+(?:me\s+)?this\s+page\b"
    r"|\bwhat\s+does\s+this\s+page\s+say\b",
    re.IGNORECASE,
)


def _active_window() -> dict | None:
    try:
        out = subprocess.run(
            ["hyprctl", "-j", "activewindow"], capture_output=True, text=True, timeout=2,
        )
        data = json.loads(out.stdout)
        return data or None
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return None


def _active_browser_title() -> str | None:
    win = _active_window()
    if not win or str(win.get("class", "")).lower() not in BROWSER_WINDOW_CLASSES:
        return None
    return win.get("title") or None


def _query_browser_extension(timeout: float = 2.0) -> dict | None:
    """Asks the Firefox extension's native-messaging host (listening on
    BROWSER_IPC_SOCKET) for the active tab's title/url/text. Returns
    None if the extension isn't installed/running or the socket read
    times out — callers should treat that as "feature unavailable"."""
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            s.connect(BROWSER_IPC_SOCKET)
            s.sendall(json.dumps({"cmd": "get_page"}).encode() + b"\n")
            chunks = []
            while True:
                chunk = s.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
                if chunk.endswith(b"\n"):
                    break
            if not chunks:
                return None
            return json.loads(b"".join(chunks).decode())
    except (OSError, json.JSONDecodeError):
        return None


def _summarize_page(title: str, url: str, text: str) -> str:
    """One-off Ollama call, deliberately kept OUT of conversation
    history — page text can be long, and there's no reason for it to
    stick around in context for unrelated later turns."""
    messages = [
        {
            "role": "system",
            "content": (
                "Summarize the given web page in 1-2 short spoken "
                "sentences. Be concise and factual, no preamble."
            ),
        },
        {"role": "user", "content": f"Title: {title}\nURL: {url}\n\nContent:\n{text[:4000]}"},
    ]
    try:
        summary = "".join(ask_ollama_stream(messages)).strip()
    except requests.exceptions.RequestException:
        return "Couldn't reach Ollama to summarize that page."
    return summary or "That page doesn't have much text to summarize."


def try_handle_browser(text: str) -> str | None:
    if BROWSER_TAB_RE.search(text):
        title = _active_browser_title()
        if title is None:
            return "I don't see a Firefox window open."
        return f"You're on {title}."

    if BROWSER_SUMMARY_RE.search(text):
        page = _query_browser_extension()
        if page is None:
            return "Can't reach the Firefox extension — check it's installed and running."
        return _summarize_page(page.get("title", ""), page.get("url", ""), page.get("text", ""))

    return None


# ----------------------------------------------------------------------
# System toggles — wifi, bluetooth, night light. Same closed-whitelist
# spirit as everything else: each phrase maps to exactly one fixed
# command, nothing free-form is ever run.
#
# EDIT THESE if your setup differs:
#   - Wi-Fi assumes NetworkManager (`nmcli`). If you're on iwd/systemd-
#     networkd directly instead, swap the nmcli calls for the
#     equivalent (e.g. `iwctl station wlan0 disconnect` / a scan-and-
#     connect script) — there's no universal "off" for those the way
#     nmcli provides.
#   - Bluetooth assumes `bluetoothctl` (bluez), which accepts `power
#     on`/`power off` as direct non-interactive arguments in recent
#     versions — confirm with `bluetoothctl power on` in a terminal
#     first if this doesn't work.
#   - Night light assumes `hyprsunset` (common on Hyprland/end-4
#     setups) with no persistent daemon by default — "on" starts it,
#     "off" kills it. Check `which hyprsunset`; if you use `gammastep`
#     or `wlsunset` instead, swap the two subprocess calls below for
#     their start/stop equivalents.
# ----------------------------------------------------------------------

NIGHTLIGHT_TEMP_K = "4000"  # warmth in Kelvin for hyprsunset; lower = warmer

WIFI_ON_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?wi-?fi\s+on\b"
    r"|\b(?:turn|switch)\s+on\s+(?:the\s+)?wi-?fi\b"
    r"|\benable\s+(?:the\s+)?wi-?fi\b",
    re.IGNORECASE,
)
WIFI_OFF_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?wi-?fi\s+off\b"
    r"|\b(?:turn|switch)\s+off\s+(?:the\s+)?wi-?fi\b"
    r"|\bdisable\s+(?:the\s+)?wi-?fi\b",
    re.IGNORECASE,
)
BLUETOOTH_ON_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?bluetooth\s+on\b"
    r"|\b(?:turn|switch)\s+on\s+(?:the\s+)?bluetooth\b"
    r"|\benable\s+(?:the\s+)?bluetooth\b",
    re.IGNORECASE,
)
BLUETOOTH_OFF_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?bluetooth\s+off\b"
    r"|\b(?:turn|switch)\s+off\s+(?:the\s+)?bluetooth\b"
    r"|\bdisable\s+(?:the\s+)?bluetooth\b",
    re.IGNORECASE,
)
NIGHTLIGHT_ON_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?night\s*light\s+on\b"
    r"|\b(?:turn|switch)\s+on\s+(?:the\s+)?night\s*light\b"
    r"|\benable\s+(?:the\s+)?night\s*light\b",
    re.IGNORECASE,
)
NIGHTLIGHT_OFF_RE = re.compile(
    r"\b(?:turn|switch)\s+(?:the\s+)?night\s*light\s+off\b"
    r"|\b(?:turn|switch)\s+off\s+(?:the\s+)?night\s*light\b"
    r"|\bdisable\s+(?:the\s+)?night\s*light\b",
    re.IGNORECASE,
)


def _process_running(name: str) -> bool:
    try:
        return subprocess.run(
            ["pgrep", "-x", name],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
        ).returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def try_handle_wifi(text: str) -> str | None:
    if WIFI_ON_RE.search(text):
        try:
            subprocess.run(
                ["nmcli", "radio", "wifi", "on"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach nmcli — check NetworkManager is running."
        return "Wi-Fi on."
    if WIFI_OFF_RE.search(text):
        try:
            subprocess.run(
                ["nmcli", "radio", "wifi", "off"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach nmcli — check NetworkManager is running."
        return "Wi-Fi off."
    return None


def try_handle_bluetooth(text: str) -> str | None:
    if BLUETOOTH_ON_RE.search(text):
        try:
            subprocess.run(
                ["bluetoothctl", "power", "on"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach bluetoothctl — check the bluetooth service is running."
        return "Bluetooth on."
    if BLUETOOTH_OFF_RE.search(text):
        try:
            subprocess.run(
                ["bluetoothctl", "power", "off"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach bluetoothctl — check the bluetooth service is running."
        return "Bluetooth off."
    return None


def try_handle_nightlight(text: str) -> str | None:
    if NIGHTLIGHT_ON_RE.search(text):
        if _process_running("hyprsunset"):
            return "Night light's already on."
        try:
            subprocess.Popen(
                ["hyprsunset", "-t", NIGHTLIGHT_TEMP_K],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return "Couldn't find hyprsunset — check it's installed."
        return "Night light on."
    if NIGHTLIGHT_OFF_RE.search(text):
        if not _process_running("hyprsunset"):
            return "Night light's already off."
        try:
            subprocess.run(
                ["pkill", "-x", "hyprsunset"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't turn off night light — pkill isn't available."
        return "Night light off."
    return None


# ----------------------------------------------------------------------
# System volume & screen brightness. Same closed-whitelist pattern as
# everything else: fixed step size, fixed commands, an optional
# explicit percentage — nothing free-form is ever run.
#
# EDIT THESE if your setup differs:
#   - Volume assumes PipeWire via `wpctl` (default on modern Arch/Athena
#     Hyprland setups — check with `wpctl status`). If you're on plain
#     PulseAudio instead, swap for `pactl set-sink-volume @DEFAULT_SINK@
#     +5%` / `pactl set-sink-mute @DEFAULT_SINK@ toggle`.
#   - Brightness assumes `brightnessctl` is installed and your user is
#     in the `video` group (needed to write sysfs without sudo). Test
#     with `brightnessctl set 50%` in a terminal first — if it needs
#     sudo, add yourself to that group and re-login rather than running
#     this script as root.
# ----------------------------------------------------------------------

VOLUME_STEP = "5%"
BRIGHTNESS_STEP = "5%"

VOLUME_SET_RE = re.compile(r"\bset\s+(?:the\s+)?volume\s+to\s+(\d{1,3})\s*(?:%|percent)?\b", re.IGNORECASE)
VOLUME_UP_RE = re.compile(r"\bvolume\s*up\b|\blouder\b|\bturn\s+(?:it|the\s+volume)\s+up\b", re.IGNORECASE)
VOLUME_DOWN_RE = re.compile(
    r"\bvolume\s*down\b|\bquieter\b|\blower\s+the\s+volume\b|\bturn\s+(?:it|the\s+volume)\s+down\b",
    re.IGNORECASE,
)
MUTE_RE = re.compile(r"\bmute\b", re.IGNORECASE)
UNMUTE_RE = re.compile(r"\bunmute\b", re.IGNORECASE)

BRIGHTNESS_SET_RE = re.compile(
    r"\bset\s+(?:the\s+)?(?:screen\s+)?brightness\s+to\s+(\d{1,3})\s*(?:%|percent)?\b", re.IGNORECASE
)
BRIGHTNESS_UP_RE = re.compile(
    r"\bbrightness\s*up\b|\bbrighten\s+(?:the\s+)?screen\b|\bturn\s+(?:it|the\s+brightness)\s+up\b",
    re.IGNORECASE,
)
BRIGHTNESS_DOWN_RE = re.compile(
    r"\bbrightness\s*down\b|\bdim\s+(?:the\s+)?screen\b|\bturn\s+(?:it|the\s+brightness)\s+down\b",
    re.IGNORECASE,
)


def try_handle_volume(text: str) -> str | None:
    """System-wide output volume via wpctl — affects everything playing,
    not just mpv. This replaces the old mpv-only volume commands, since
    'volume up' normally means the whole machine, not just whatever
    Nyx happens to be playing at the time."""
    match = VOLUME_SET_RE.search(text)
    if match:
        pct = max(0, min(100, int(match.group(1))))
        try:
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{pct}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach wpctl — check PipeWire's running."
        return f"Volume set to {pct} percent."

    if MUTE_RE.search(text):
        try:
            subprocess.run(
                ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "1"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach wpctl — check PipeWire's running."
        return "Muted."
    if UNMUTE_RE.search(text):
        try:
            subprocess.run(
                ["wpctl", "set-mute", "@DEFAULT_AUDIO_SINK@", "0"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach wpctl — check PipeWire's running."
        return "Unmuted."

    if VOLUME_UP_RE.search(text):
        try:
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{VOLUME_STEP}+"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach wpctl — check PipeWire's running."
        return "Volume up."
    if VOLUME_DOWN_RE.search(text):
        try:
            subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", f"{VOLUME_STEP}-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach wpctl — check PipeWire's running."
        return "Volume down."
    return None


def try_handle_brightness(text: str) -> str | None:
    """Screen brightness via brightnessctl."""
    match = BRIGHTNESS_SET_RE.search(text)
    if match:
        pct = max(0, min(100, int(match.group(1))))
        try:
            subprocess.run(
                ["brightnessctl", "set", f"{pct}%"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach brightnessctl — check it's installed."
        return f"Brightness set to {pct} percent."

    if BRIGHTNESS_UP_RE.search(text):
        try:
            subprocess.run(
                ["brightnessctl", "set", f"{BRIGHTNESS_STEP}+"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach brightnessctl — check it's installed."
        return "Brightness up."
    if BRIGHTNESS_DOWN_RE.search(text):
        try:
            subprocess.run(
                ["brightnessctl", "set", f"{BRIGHTNESS_STEP}-"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=5,
            )
        except (subprocess.SubprocessError, FileNotFoundError):
            return "Couldn't reach brightnessctl — check it's installed."
        return "Brightness down."
    return None


def try_handle_tools(text: str) -> str | None:
    """Keyword-routed tools. Returns a spoken reply if a tool handled the
    command, or None to fall through to the LLM for a normal chat reply."""
    # Checked first: "open spotify and play X" / "play X on spotify"
    # both contain "play ...", which the generic PLAY_MUSIC_RE below
    # would otherwise grab first and route to the local mpv player.
    spotify_reply = try_handle_spotify(text)
    if spotify_reply is not None:
        return spotify_reply

    open_reply = try_handle_open(text)
    if open_reply is not None:
        return open_reply

    close_reply = try_handle_close(text)
    if close_reply is not None:
        return close_reply

    wallpaper_reply = try_handle_wallpaper(text)
    if wallpaper_reply is not None:
        return wallpaper_reply

    browser_reply = try_handle_browser(text)
    if browser_reply is not None:
        return browser_reply

    search_reply = try_handle_search(text)
    if search_reply is not None:
        return search_reply

    datetime_reply = try_handle_datetime(text)
    if datetime_reply is not None:
        return datetime_reply

    wifi_reply = try_handle_wifi(text)
    if wifi_reply is not None:
        return wifi_reply

    bluetooth_reply = try_handle_bluetooth(text)
    if bluetooth_reply is not None:
        return bluetooth_reply

    nightlight_reply = try_handle_nightlight(text)
    if nightlight_reply is not None:
        return nightlight_reply

    volume_reply = try_handle_volume(text)
    if volume_reply is not None:
        return volume_reply

    brightness_reply = try_handle_brightness(text)
    if brightness_reply is not None:
        return brightness_reply

    match = PLAY_MUSIC_RE.search(text)
    if match:
        query = match.group(1).strip()
        if query:
            play_music(query)
            return quip("play_music", f"Playing {query}.", query=query)

    if STOP_RE.search(text):
        return quip("stop_music", "Stopped.") if stop_music() else "Nothing's playing."
    if PAUSE_RE.search(text):
        return "Paused." if mpv_command("set_property", "pause", True) else "Nothing's playing."
    if RESUME_RE.search(text):
        return "Resuming." if mpv_command("set_property", "pause", False) else "Nothing's playing."
    if SKIP_RE.search(text):
        # A single ytsearch stream has no queued "next" track, so skipping
        # just stops the current one. Wire up a real playlist if you want
        # actual skip-to-next behavior.
        return "Skipping." if stop_music() else "Nothing's playing."

    return None


# ----------------------------------------------------------------------
# Conversation handling
# ----------------------------------------------------------------------

def respond(
    text: str,
    history: list[dict],
    voice: PiperVoice,
    audio_q: queue.Queue,
    ww_model: WakeWordModel,
) -> bool:
    """Handle one user utterance: tools or a streamed LLM reply + TTS.
    Returns True if playback was cut off by a barge-in wake word."""
    tool_reply = try_handle_tools(text)
    interrupted = False

    if tool_reply is not None:
        print(f"Nyx:    {tool_reply}")
        # Tagged "tool": True so this line is spoken/saved/logged normally
        # but never fed back into the LLM's own context — otherwise it
        # reads its own past "Playing X." replies as precedent and starts
        # imitating them in unrelated turns (this is what was happening).
        history.append({"role": "assistant", "content": tool_reply, "tool": True})
        interrupted = speak_interruptible(voice, tool_reply, audio_q, ww_model)
    else:
        parts: list[str] = []
        try:
            for sentence in stream_sentences(ask_ollama_stream(llm_messages(history))):
                parts.append(sentence)
                print(f"Nyx:    {sentence}")
                if speak_interruptible(voice, sentence, audio_q, ww_model):
                    interrupted = True
                    break
                if len(parts) >= MAX_REPLY_SENTENCES:
                    # Model kept going past a normal reply length — stop
                    # here rather than speak a runaway/hallucinated tail.
                    break
        except requests.exceptions.RequestException as e:
            print(f"Ollama error: {e}", file=sys.stderr)
            if not parts:
                fallback = "I can't reach Ollama right now."
                print(f"Nyx:    {fallback}")
                interrupted = speak_interruptible(voice, fallback, audio_q, ww_model)
                parts = [fallback]
        history.append({"role": "assistant", "content": " ".join(parts)})

    trim_history(history)
    return interrupted


def run_conversation(
    audio_q: queue.Queue,
    whisper: WhisperModel,
    voice: PiperVoice,
    history: list[dict],
    ww_model: WakeWordModel,
    vad: webrtcvad.Vad,
) -> None:
    """Handles one command triggered by the wake word. No open-mic
    follow-up window — Nyx never listens without an explicit wake word
    said first, which is what was causing Whisper to hallucinate whole
    fake exchanges out of silence/room noise during that idle window.
    The only case this loops back to listening on its own is a barge-in:
    if the wake word is said again *while Nyx is still talking*, that's
    itself a fresh, explicit wake event, so it goes straight back into
    listening rather than requiring the phrase to be repeated once more
    after playback stops.
    """
    print("Wake word detected — listening...")
    overlay_ipc.send_state("listening")
    play_earcon()

    while True:
        drain_queue(audio_q)
        audio = record_command(
            audio_q, vad, initial_timeout=MAX_RECORD_SECONDS, max_total=MAX_RECORD_SECONDS
        )
        if audio.size == 0:
            return

        overlay_ipc.send_state("thinking")
        text = transcribe(whisper, audio)
        if not text:
            return

        print(f"You:    {text}")
        history.append({"role": "user", "content": text})

        interrupted = respond(text, history, voice, audio_q, ww_model)
        if not interrupted:
            return
        print("(interrupted — listening...)")
        overlay_ipc.send_state("listening")


# ----------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------

def main() -> None:
    print("Loading wake word model...")
    model_path = WAKE_WORD_MODEL_PATH or openwakeword.models["hey_jarvis"]["model_path"]
    ww_model = WakeWordModel(wakeword_model_paths=[model_path])

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

    print(f"Loading Whisper ({WHISPER_MODEL_SIZE}, {WHISPER_DEVICE})...")
    try:
        whisper = WhisperModel(
            WHISPER_MODEL_SIZE, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE
        )
    except Exception as e:
        print(f"GPU load failed ({e}); falling back to CPU.", file=sys.stderr)
        whisper = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")

    print("Loading Piper voice...")
    voice = PiperVoice.load(PIPER_MODEL_PATH)

    history = load_history()
    print(f"Loaded {len(history) - 1} saved message(s) from history.")

    audio_q: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_q.put(bytes(indata))

    overlay_ipc.start()
    overlay_ipc.send_state("idle")

    print(f'Ready. Say "{WAKE_PHRASE}" to wake Nyx.')
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SAMPLES,
        dtype="int16",
        channels=1,
        callback=callback,
    ):
        while True:
            frame = np.frombuffer(audio_q.get(), dtype=np.int16)
            prediction = ww_model.predict(frame)
            score = prediction.get(WAKE_WORD_KEY, 0.0)
            if score > 0.05:  # DEBUG — remove once wake word is working reliably
                print(f"[debug] wake score: {score:.3f}")
            if score > WAKE_THRESHOLD:
                ww_model.reset()
                run_conversation(audio_q, whisper, voice, history, ww_model, vad)
                save_history(history)
                overlay_ipc.send_state("idle")
                print(f'\nReady. Say "{WAKE_PHRASE}" to wake Nyx.')


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nShutting down.")
