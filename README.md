# Nyx

A fully local, offline, GPU-accelerated voice assistant for **Athena OS (Arch-based) + Hyprland** with the **[end-4 dots-hyprland](https://github.com/end-4/dots-hyprland)** dotfiles (quickshell `ii`).

Say **"Hey Nyx"**, talk, and Nyx answers out loud. No cloud calls, no API keys (the optional Spotify lookup uses your own free developer credentials).

```
mic → openWakeWord → VAD-gated recording → faster-whisper (STT)
    → Ollama (streamed) → Piper (TTS, sentence by sentence) → speakers
```

## Features

- **Custom wake word** – `hey_nyx.onnx` is included. Barge-in: say the wake word while Nyx is talking and it stops and listens.
- **Streaming replies** – Piper starts speaking each sentence as soon as Ollama produces it.
- **WebRTC VAD** – adaptive silence detection instead of a fixed volume threshold.
- **Persistent memory** – chat history is saved to `~/.local/share/nyx-assistant/history.json`.
- **Closed-whitelist tools** – Nyx never runs arbitrary commands. Unrecognised phrases just fall through to normal chat.
- **Orb overlay** – pushes state (idle / listening / thinking / speaking) and audio level to a quickshell widget over a Unix socket (`overlay_ipc.py`). Optional: it is a silent no-op if the widget isn't running.

### Voice commands

| Category | Examples | Uses |
|---|---|---|
| Music (YouTube) | "play <song>", "pause", "resume", "skip", "stop" | `mpv` + `yt-dlp` |
| Spotify | "open spotify and play <song>" | Spotify desktop app via D-Bus |
| Apps / sites / folders | "open Firefox", "open YouTube", "open Downloads" | `xdg-open` |
| Close apps | "close the browser" | `hyprctl` |
| Web search | "search for <query>" | opens a Google search |
| Date / time | "what time is it" | system clock |
| Wi-Fi | "turn wifi on/off" | `nmcli` |
| Bluetooth | "turn bluetooth on/off" | `bluetoothctl` |
| Night light | "turn night light on/off" | `hyprsunset` |
| System volume | "volume up/down", "mute", "set volume to 40 percent" | `wpctl` |
| Brightness | "brightness up/down", "set brightness to 60 percent" | `brightnessctl` |
| Wallpaper | "change wallpaper", "random wallpaper", "change wallpaper to <name>" | end-4's `switchwall.sh` |
| Browser page | "what tab am I on", "summarize this page" | Firefox + optional extension |

## Requirements

- Athena OS / Arch Linux, Hyprland, end-4 dots-hyprland (quickshell `ii`)
- NVIDIA GPU recommended (Whisper runs on CUDA). Falls back to CPU automatically.
- PipeWire + WirePlumber
- Python 3.10+

### System packages

```fish
sudo pacman -S --needed python python-pip fish git base-devel portaudio \
    mpv yt-dlp ollama-cuda networkmanager bluez-utils brightnessctl \
    hyprsunset wireplumber xdg-utils
```

Spotify support also needs the Spotify desktop app (AUR: `yay -S spotify`).
Drop `ollama-cuda` for `ollama` if you have no NVIDIA GPU.

## Install

```fish
git clone https://github.com/abhishek-lore/nyx-assistant.git
cd nyx-assistant

python -m venv venv
source venv/bin/activate.fish
pip install -r requirements.txt

# CUDA libs that faster-whisper needs (run.fish adds them to LD_LIBRARY_PATH)
pip install nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

> If `webrtcvad` fails to build, use `pip install webrtcvad-wheels` instead.

### LLM (Ollama)

```fish
sudo systemctl enable --now ollama
ollama pull gemma2:2b-instruct-q4_K_M
```

Any model works, just set `OLLAMA_MODEL` in `main.py` to the exact tag from `ollama list`.

### Piper voice

Voice models are large, so they are not committed to git. Download one into `voices/`:

```fish
mkdir -p voices
python -m piper.download_voices en_US-lessac-medium --download-dir voices
```

Alternatively, grab the `.onnx` and `.onnx.json` pair from [rhasspy/piper-voices](https://huggingface.co/rhasspy/piper-voices) and place both files in `voices/`. `PIPER_MODEL_PATH` in `main.py` must point at it.

### Spotify (optional)

1. Create a free app at <https://developer.spotify.com/dashboard> (no Redirect URI needed).
2. Run `python spotify_setup.py` and paste the Client ID and Secret.

Credentials are saved to `~/.config/nyx-assistant/spotify.json` (mode `600`). Nyx only uses them to look up tracks; playback happens in your running Spotify app.

## Run

```fish
chmod +x run.fish
./run.fish
```

`run.fish` activates the venv, exports the CUDA library paths, and starts `main.py`.

### Autostart with Hyprland (end-4)

Add this to `~/.config/hypr/custom/execs.conf` (or `hyprland.conf` on older versions):

```ini
exec-once = fish /path/to/nyx-assistant/run.fish
```

## Configuration

Everything is at the top of `main.py`:

| Setting | Purpose |
|---|---|
| `WAKE_WORD_MODEL_PATH`, `WAKE_WORD_KEY`, `WAKE_THRESHOLD` | wake word model and sensitivity |
| `OLLAMA_MODEL`, `OLLAMA_MAX_TOKENS` | LLM tag and reply length |
| `WHISPER_MODEL_SIZE`, `WHISPER_DEVICE`, `WHISPER_COMPUTE` | STT (use `cpu` / `int8` without CUDA) |
| `PIPER_MODEL_PATH` | TTS voice |
| `AUDIO_INPUT_DEVICE` | pinned to `"pipewire"`; check `python -c "import sounddevice as sd; print(sd.query_devices())"` if the mic goes silent |
| `VAD_AGGRESSIVENESS`, `SILENCE_HANG_FRAMES` | end-of-speech detection |
| `WALLPAPER_DIR`, `SWITCHWALL_SCRIPT`, `RANDOM_WALLPAPER_SCRIPT` | end-4 wallpaper integration (`~/.config/quickshell/ii/scripts/colors/`) |
| `NIGHTLIGHT_TEMP_K`, `VOLUME_STEP`, `BRIGHTNESS_STEP` | system toggle defaults |

## Project layout

```
nyx-assistant/
├── main.py            # the assistant
├── overlay_ipc.py     # state/level socket for the quickshell orb
├── spotify_setup.py   # one-time Spotify credential setup
├── run.fish           # launcher (venv + CUDA paths)
├── requirements.txt
├── hey_nyx.onnx       # custom wake word model
├── quickshell/NyxOrb.qml  # orb widget (copy from ~/.config/quickshell/ii/modules/nyxOrb/)
└── voices/            # Piper voices (downloaded, git-ignored)
```

## Troubleshooting

- **CUDA / cuDNN errors** – make sure the two `nvidia-*` pip packages are installed, and launch via `run.fish`. Otherwise set `WHISPER_DEVICE = "cpu"`, `WHISPER_COMPUTE = "int8"`.
- **Wake word doesn't trigger** – watch the `[debug] wake score` lines in the terminal and adjust `WAKE_THRESHOLD`.
- **No mic input** – re-check `AUDIO_INPUT_DEVICE` (see Configuration).
- **Wallpaper commands fail** – confirm the end-4 script paths exist and that `switchwall.sh /path/to/image.jpg` works in a terminal.
- **Orb not moving** – Nyx creates `/tmp/nyx-overlay.sock`; the widget must be running inside your quickshell config (`qs -c ii`). Install it by copying `NyxOrb.qml` to `~/.config/quickshell/ii/modules/nyxOrb/`, then import and instantiate it in `shell.qml`.

## License


[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This project is licensed under the MIT License.

---
