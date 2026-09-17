"""
AI Instrument Separator (Demucs)
=========================================================
Desktop GUI application that allows you to:
- Drag and drop an audio file (or select it manually)
- Choose a "parent" folder where all separations will be saved
- Choose per song between three modes:
    * 6 instruments (vocals, drums, bass, guitar, piano, other)
      using htdemucs_6s - fast
    * 4 instruments in High Quality (vocals, drums, bass, other)
      using an ENSEMBLE (average) of htdemucs + htdemucs_ft + mdx_extra,
      which reduces artifacts/"muffled" sound at the cost of more
      processing time and without separating guitar/piano
    * 6 instruments in High Quality (hybrid): vocals/drums/bass come
      from the 3-model ensemble (same as the previous mode), and
      guitar/piano/other come from htdemucs_6s run with high shifts
      and overlap for better precision - the slowest mode but the
      cleanest one available for the full 6 stems
- Save the results in a subfolder "Instruments <SongName> (4/6 stems)"
  inside the chosen parent folder
- Built-in mixer: play all instruments AT THE SAME TIME,
  with an independent volume slider per instrument (and a Mute button),
  plus an overall master volume, so you can build your own mix
  or isolate/remove instruments live
- A YouTube-style seek bar to scrub through the currently loaded mix
- Detect and display whether GPU (CUDA) or CPU is being used

Requirements (install before running, see requirements.txt):
    pip install -r requirements.txt

Usage:
    python separator.py
"""

import os
import sys
import threading
import queue
import subprocess
import shutil
import tempfile
from pathlib import Path

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

import pygame
import numpy as np

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# ----------------------------------------------------------------------
# General configuration
# ----------------------------------------------------------------------

MODEL_6STEM = "htdemucs_6s"
ENSEMBLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra"]  # averaged together

STEM_NAMES_6 = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]

STEM_LABELS = {
    "vocals": "Vocals",
    "drums": "Drums",
    "bass": "Bass",
    "guitar": "Guitar",
    "piano": "Piano",
    "other": "Other",
}

APP_TITLE = "AI Instrument Separator (Demucs)"

MODE_6STEM = "6stem"
MODE_ENSEMBLE = "ensemble"
MODE_HYBRID_6STEM = "hybrid6"

# "shifts" and "overlap" parameters used to improve htdemucs_6s precision
# in hybrid mode (slower, but cleaner for guitar/piano/other)
HYBRID_SHIFTS = 5
HYBRID_OVERLAP = 0.5

# ----------------------------------------------------------------------
# Color palette (modern, dark, based on #020e21)
# ----------------------------------------------------------------------

COLOR_BG = "#020e21"          # main background
COLOR_BG_ALT = "#071733"      # slightly lighter panels
COLOR_BG_CARD = "#0b1f42"     # cards / framed sections
COLOR_BORDER = "#16305c"      # subtle borders
COLOR_TEXT = "#e7edf7"        # primary text
COLOR_TEXT_MUTED = "#8ea2c4"  # secondary text
COLOR_ACCENT = "#3b82f6"      # accent blue
COLOR_ACCENT_DARK = "#2563eb"
COLOR_SUCCESS = "#22c55e"
COLOR_ERROR = "#ef4444"
COLOR_TRACK = "#132a52"       # slider track background


# ----------------------------------------------------------------------
# GPU detection utilities
# ----------------------------------------------------------------------

def get_device_info():
    if not TORCH_AVAILABLE:
        return "PyTorch not installed (cannot check GPU)."
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU detected: {name} (CUDA will be used automatically)"
        else:
            return "No CUDA-compatible GPU detected. CPU will be used (slower)."
    except Exception as e:
        return f"Could not check device: {e}"


# ----------------------------------------------------------------------
# Separation logic (runs on a separate thread so the UI doesn't freeze)
# ----------------------------------------------------------------------

def _make_safe_copy(input_file, temp_dir):
    """
    Copies the input file to a "safe" filename inside temp_dir, avoiding
    problematic characters on Windows (trailing dots, exclamation marks,
    quotes, etc. often cause Windows to truncate or reject the folder
    that demucs tries to create with that same name).
    Returns the path to the copied file.
    """
    ext = Path(input_file).suffix
    safe_name = f"input_audio{ext}"
    safe_path = Path(temp_dir) / safe_name
    shutil.copy(str(input_file), str(safe_path))
    return safe_path


def _run_single_demucs(model_name, input_file, work_dir, log_queue, extra_args=None):
    """Runs demucs for a single model and returns the folder containing the wavs."""
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model_name,
        "-o", str(work_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(input_file))

    log_queue.put(("log", f"--- Running model: {model_name} {' '.join(extra_args or [])} ---"))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        log_queue.put(("log", line.rstrip()))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Demucs failed with model {model_name} (exit code {process.returncode})")

    song_name = Path(input_file).stem
    out_dir = Path(work_dir) / model_name / song_name
    if not out_dir.exists():
        raise RuntimeError(f"Expected output folder was not created for {model_name}: {out_dir}")
    return out_dir


def _average_wavs(paths, destination):
    """Averages a list of .wav files (expected to have the same duration) and saves the result."""
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("Missing 'soundfile' library to average audio (pip install soundfile).")

    data_list = []
    samplerate = None
    for p in paths:
        data, sr = sf.read(str(p))
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise RuntimeError(f"Different sample rates between models: {sr} vs {samplerate}")
        data_list.append(data)

    min_len = min(d.shape[0] for d in data_list)
    data_list = [d[:min_len] for d in data_list]

    averaged = np.mean(np.stack(data_list, axis=0), axis=0)
    sf.write(str(destination), averaged, samplerate)


def run_separation(input_file: str, output_root: str, mode: str, log_queue: "queue.Queue"):
    """
    Runs the separation according to the chosen mode:
    - MODE_6STEM: a single model (htdemucs_6s), 6 stems.
    - MODE_ENSEMBLE: runs 3 four-stem models and averages the results
      instrument by instrument to reduce artifacts.
    - MODE_HYBRID_6STEM: 6 stems combining both modes **VERY EXPERIMENTAL**:
        * vocals/drums/bass -> average (ensemble) of the 3 four-stem models
        * guitar/piano/other -> htdemucs_6s run with high shifts+overlap
          (more precise), since it's the only model that can reliably
          separate guitar and piano from "other"
    """
    song_name = Path(input_file).stem
    stems_label = {
        MODE_6STEM: "6 stems",
        MODE_ENSEMBLE: "4 stems",
        MODE_HYBRID_6STEM: "6 stems HQ",
    }.get(mode, "stems")
    final_folder = Path(output_root) / f"Instruments {song_name} ({stems_label})"
    final_folder.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="demucs_input_") as safe_input_dir:
            safe_input = _make_safe_copy(input_file, safe_input_dir)
            log_queue.put(("log", f"Safe copy of the input file: {safe_input.name}"))

            if mode == MODE_6STEM:
                log_queue.put(("status", f"Separating '{song_name}' into 6 instruments ({MODEL_6STEM})..."))
                out_dir = _run_single_demucs(MODEL_6STEM, safe_input, output_root, log_queue)

                moved_files = {}
                for stem in STEM_NAMES_6:
                    src = out_dir / f"{stem}.wav"
                    if src.exists():
                        dst = final_folder / f"{stem}.wav"
                        shutil.move(str(src), str(dst))
                        moved_files[stem] = str(dst)
                    else:
                        log_queue.put(("log", f"Warning: {stem}.wav was not generated"))

                _cleanup_model_dir(output_root, MODEL_6STEM, out_dir)
                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            elif mode == MODE_ENSEMBLE:
                log_queue.put((
                    "status",
                    f"Separating '{song_name}' in High Quality (ensemble of {len(ENSEMBLE_MODELS)} models)... "
                    "This will take longer than normal mode."
                ))

                with tempfile.TemporaryDirectory(prefix="ensemble_work_") as tmp_root:
                    model_out_dirs = []
                    for model_name in ENSEMBLE_MODELS:
                        out_dir = _run_single_demucs(model_name, safe_input, tmp_root, log_queue)
                        model_out_dirs.append(out_dir)

                    log_queue.put(("status", "Averaging model results to reduce artifacts..."))

                    moved_files = {}
                    for stem in STEM_NAMES_4:
                        stem_paths = [d / f"{stem}.wav" for d in model_out_dirs]
                        stem_paths = [p for p in stem_paths if p.exists()]
                        if not stem_paths:
                            log_queue.put(("log", f"Warning: no model generated {stem}.wav"))
                            continue
                        dst = final_folder / f"{stem}.wav"
                        if len(stem_paths) == 1:
                            shutil.copy(str(stem_paths[0]), str(dst))
                        else:
                            _average_wavs(stem_paths, dst)
                        moved_files[stem] = str(dst)
                        log_queue.put(("log", f"{stem}: averaged from {len(stem_paths)} models -> {dst.name}"))

                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            elif mode == MODE_HYBRID_6STEM:
                log_queue.put((
                    "status",
                    f"Separating '{song_name}' into 6 High Quality instruments "
                    f"(vocals/drums/bass via ensemble + guitar/piano/other via {MODEL_6STEM} with shifts)... "
                    "This is the slowest mode."
                ))

                moved_files = {}

                # --- Part 1: vocals/drums/bass via ensemble of 3 four-stem models ---
                with tempfile.TemporaryDirectory(prefix="hybrid_ensemble_") as tmp_root:
                    model_out_dirs = []
                    for model_name in ENSEMBLE_MODELS:
                        out_dir = _run_single_demucs(model_name, safe_input, tmp_root, log_queue)
                        model_out_dirs.append(out_dir)

                    log_queue.put(("status", "Averaging vocals/drums/bass across the 3 models..."))
                    for stem in ["vocals", "drums", "bass"]:
                        stem_paths = [d / f"{stem}.wav" for d in model_out_dirs]
                        stem_paths = [p for p in stem_paths if p.exists()]
                        if not stem_paths:
                            log_queue.put(("log", f"Warning: no model generated {stem}.wav"))
                            continue
                        dst = final_folder / f"{stem}.wav"
                        if len(stem_paths) == 1:
                            shutil.copy(str(stem_paths[0]), str(dst))
                        else:
                            _average_wavs(stem_paths, dst)
                        moved_files[stem] = str(dst)
                        log_queue.put(("log", f"{stem}: averaged from {len(stem_paths)} models -> {dst.name}"))

                # --- Part 2: guitar/piano/other via htdemucs_6s with shifts+overlap ---
                with tempfile.TemporaryDirectory(prefix="hybrid_6s_") as tmp_root2:
                    log_queue.put((
                        "status",
                        f"Separating guitar/piano/other with {MODEL_6STEM} "
                        f"(shifts={HYBRID_SHIFTS}, overlap={HYBRID_OVERLAP})..."
                    ))
                    extra_args = ["--shifts", str(HYBRID_SHIFTS), "--overlap", str(HYBRID_OVERLAP)]
                    out_dir_6s = _run_single_demucs(
                        MODEL_6STEM, safe_input, tmp_root2, log_queue, extra_args=extra_args
                    )
                    for stem in ["guitar", "piano", "other"]:
                        src = out_dir_6s / f"{stem}.wav"
                        if src.exists():
                            dst = final_folder / f"{stem}.wav"
                            shutil.copy(str(src), str(dst))
                            moved_files[stem] = str(dst)
                        else:
                            log_queue.put(("log", f"Warning: {stem}.wav was not generated"))

                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            else:
                log_queue.put(("error", f"Unknown mode: {mode}"))

    except FileNotFoundError:
        log_queue.put(("error", "'demucs' was not found. Install it with: pip install demucs"))
    except Exception as e:
        log_queue.put(("error", f"Unexpected error: {e}"))


def _cleanup_model_dir(output_root, model_name, out_dir):
    try:
        shutil.rmtree(out_dir)
        model_dir = Path(output_root) / model_name
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Graphical interface
# ----------------------------------------------------------------------

class SeparatorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("820x760")
        self.root.minsize(480, 420)
        self.root.configure(bg=COLOR_BG)

        self.output_root = tk.StringVar(value=str(Path.home() / "AI_Separations"))
        self.input_file = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Waiting for an audio file...")
        self.volume = tk.DoubleVar(value=70.0)
        self.mode = tk.StringVar(value=MODE_6STEM)

        self.log_queue = queue.Queue()
        self.current_stem_files = {}
        self.current_stem_list = STEM_NAMES_6

        # Playback / seek state
        self.track_length_sec = 0.0
        self.is_playing = False
        self.is_seeking = False
        self.playback_offset_sec = 0.0   # position (sec) at which playback last started
        self.playback_started_at = None  # time.time() when playback last started

        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)  # enough channels for 6 simultaneous stems

        self._setup_style()
        self._build_scrollable_root()
        self._build_ui()
        self._poll_queue()
        self._tick_seekbar()

    # ---------------------------------------------------------- ttk theme
    def _setup_style(self):
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT,
                         font=("Segoe UI", 10))
        style.configure("TFrame", background=COLOR_BG)
        style.configure("Card.TFrame", background=COLOR_BG_CARD)
        style.configure("TLabel", background=COLOR_BG, foreground=COLOR_TEXT)
        style.configure("Card.TLabel", background=COLOR_BG_CARD, foreground=COLOR_TEXT)
        style.configure("Muted.TLabel", background=COLOR_BG_CARD, foreground=COLOR_TEXT_MUTED)
        style.configure("Header.TLabel", background=COLOR_BG, foreground=COLOR_TEXT,
                         font=("Segoe UI Semibold", 11))
        style.configure("Status.TLabel", background=COLOR_BG_CARD, foreground=COLOR_ACCENT)

        style.configure("TLabelframe", background=COLOR_BG, bordercolor=COLOR_BORDER,
                         darkcolor=COLOR_BG, lightcolor=COLOR_BG, relief="solid")
        style.configure("TLabelframe.Label", background=COLOR_BG, foreground=COLOR_TEXT_MUTED,
                         font=("Segoe UI Semibold", 9))

        style.configure("TButton", background=COLOR_BG_CARD, foreground=COLOR_TEXT,
                         borderwidth=0, focusthickness=0, padding=(14, 8))
        style.map("TButton",
                  background=[("active", COLOR_ACCENT_DARK), ("disabled", COLOR_BG_ALT)],
                  foreground=[("disabled", COLOR_TEXT_MUTED)])

        style.configure("Accent.TButton", background=COLOR_ACCENT, foreground="#ffffff",
                         borderwidth=0, padding=(16, 10), font=("Segoe UI Semibold", 10))
        style.map("Accent.TButton",
                  background=[("active", COLOR_ACCENT_DARK), ("disabled", COLOR_BG_ALT)],
                  foreground=[("disabled", COLOR_TEXT_MUTED)])

        style.configure("TRadiobutton", background=COLOR_BG, foreground=COLOR_TEXT,
                         indicatorcolor=COLOR_BG_CARD)
        style.map("TRadiobutton", background=[("active", COLOR_BG)])

        style.configure("TCheckbutton", background=COLOR_BG_CARD, foreground=COLOR_TEXT)
        style.map("TCheckbutton", background=[("active", COLOR_BG_CARD)])

        style.configure("TEntry", fieldbackground=COLOR_BG_ALT, foreground=COLOR_TEXT,
                         insertcolor=COLOR_TEXT, borderwidth=0, padding=8)

        style.configure("Horizontal.TProgressbar", background=COLOR_ACCENT,
                         troughcolor=COLOR_BG_ALT, borderwidth=0, thickness=8)

        style.configure("TScale", background=COLOR_BG_CARD, troughcolor=COLOR_TRACK,
                         borderwidth=0)
        style.configure("Seek.Horizontal.TScale", background=COLOR_BG_CARD,
                         troughcolor=COLOR_TRACK, borderwidth=0)

        style.configure("Vertical.TScrollbar", background=COLOR_BG_ALT,
                         troughcolor=COLOR_BG, borderwidth=0, arrowcolor=COLOR_TEXT_MUTED)

    # ---------------------------------------------------- scrollable root
    def _build_scrollable_root(self):
        """
        Wraps the whole UI in a canvas + scrollbar so that when the window
        is small, the user can scroll down to see everything instead of
        content being cut off.
        """
        outer = tk.Frame(self.root, bg=COLOR_BG)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, bg=COLOR_BG, highlightthickness=0)
        vscroll = ttk.Scrollbar(outer, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=vscroll.set)

        vscroll.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        self.scroll_frame = ttk.Frame(self.canvas)
        self.scroll_window = self.canvas.create_window((0, 0), window=self.scroll_frame, anchor="nw")

        self.scroll_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        )
        self.canvas.bind(
            "<Configure>",
            lambda e: self.canvas.itemconfig(self.scroll_window, width=e.width)
        )

        # Mouse wheel scrolling (Windows/Mac/Linux)
        self.canvas.bind_all("<MouseWheel>", self._on_mousewheel)
        self.canvas.bind_all("<Button-4>", self._on_mousewheel)
        self.canvas.bind_all("<Button-5>", self._on_mousewheel)

    def _on_mousewheel(self, event):
        if getattr(event, "num", None) == 4:
            self.canvas.yview_scroll(-3, "units")
        elif getattr(event, "num", None) == 5:
            self.canvas.yview_scroll(3, "units")
        else:
            self.canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    # ---------------------------------------------------------- UI build
    def _build_ui(self):
        pad = {"padx": 14, "pady": 7}
        root = self.scroll_frame

        # -- Device info banner --
        device_frame = tk.Frame(root, bg=COLOR_BG_CARD)
        device_frame.pack(fill="x", padx=14, pady=(14, 0))
        tk.Label(
            device_frame, text=get_device_info(), bg=COLOR_BG_CARD,
            fg=COLOR_SUCCESS, font=("Segoe UI", 9, "bold"), anchor="w", pady=8, padx=10
        ).pack(fill="x")

        # -- 1. Output folder --
        frame_out = ttk.LabelFrame(root, text="1.  Parent folder for separations")
        frame_out.pack(fill="x", **pad)

        entry_out = ttk.Entry(frame_out, textvariable=self.output_root)
        entry_out.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=10)
        ttk.Button(frame_out, text="Choose folder...", command=self.choose_output_folder).pack(
            side="left", padx=(0, 10), pady=10
        )

        # -- 2. Drop zone --
        frame_drop = ttk.LabelFrame(root, text="2.  Drag your song here (or click to browse)")
        frame_drop.pack(fill="x", **pad)

        self.drop_label = tk.Label(
            frame_drop,
            text="Drop your audio file here\n(MP3, WAV, FLAC, M4A...)",
            bg=COLOR_BG_ALT, fg=COLOR_TEXT,
            height=4,
            font=("Segoe UI", 11),
            relief="flat",
            cursor="hand2",
        )
        self.drop_label.pack(fill="x", padx=10, pady=10)
        self.drop_label.bind("<Button-1>", lambda e: self.choose_input_file())

        if DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop_label.config(
                text="tkinterdnd2 not installed: click here to choose the file\n"
                     "(install tkinterdnd2 to enable drag and drop)"
            )

        self.file_label = ttk.Label(frame_drop, text="No file selected.", style="Muted.TLabel")
        self.file_label.pack(padx=10, pady=(0, 10), anchor="w")

        # -- 3. Mode selection --
        frame_mode = ttk.LabelFrame(root, text="3.  Separation mode for this song")
        frame_mode.pack(fill="x", **pad)

        ttk.Radiobutton(
            frame_mode,
            text="6 instruments (vocals, drums, bass, guitar, piano, other) - fastest",
            variable=self.mode, value=MODE_6STEM,
        ).pack(anchor="w", padx=10, pady=(10, 3))

        ttk.Radiobutton(
            frame_mode,
            text="4 instruments in High Quality (vocals, drums, bass, other) - 3-model ensemble,\n"
                 "reduces \"muffled\"/artifact sound but is slower and does not separate guitar or piano",
            variable=self.mode, value=MODE_ENSEMBLE,
        ).pack(anchor="w", padx=10, pady=(3, 8))

        ttk.Radiobutton(
            frame_mode,
            text="6 instruments in High Quality (vocals/drums/bass via ensemble + guitar/piano/other\n"
                 "with shifts) - the cleanest option for all 6 stems, but also the slowest",
            variable=self.mode, value=MODE_HYBRID_6STEM,
        ).pack(anchor="w", padx=10, pady=(3, 10))

        # -- Action row --
        frame_action = ttk.Frame(root)
        frame_action.pack(fill="x", **pad)
        self.separate_btn = ttk.Button(
            frame_action, text="Separate instruments", style="Accent.TButton",
            command=self.start_separation
        )
        self.separate_btn.pack(side="left")

        self.progress = ttk.Progressbar(frame_action, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=12)

        # -- Status / log --
        frame_status = ttk.LabelFrame(root, text="Status")
        frame_status.pack(fill="x", **pad)
        ttk.Label(frame_status, textvariable=self.status_text, wraplength=760,
                  style="Status.TLabel").pack(anchor="w", padx=10, pady=(10, 5))

        self.log_box = tk.Text(
            frame_status, height=6, state="disabled",
            bg=COLOR_BG_ALT, fg=COLOR_TEXT_MUTED, relief="flat",
            insertbackground=COLOR_TEXT, padx=10, pady=8, font=("Consolas", 9)
        )
        self.log_box.pack(fill="both", padx=10, pady=(0, 10))

        # -- 4. Mixer / player --
        frame_player = ttk.LabelFrame(root, text="4.  Mixer: listen to and blend the separated instruments")
        frame_player.pack(fill="both", expand=True, **pad)

        transport_frame = ttk.Frame(frame_player)
        transport_frame.pack(fill="x", padx=10, pady=(10, 0))
        self.play_all_btn = ttk.Button(
            transport_frame, text="\u25B6  Play mix", state="disabled", command=self.play_all
        )
        self.play_all_btn.pack(side="left")
        ttk.Button(transport_frame, text="\u25A0  Stop", command=self.stop_playback).pack(
            side="left", padx=8
        )

        # -- YouTube-style seek bar --
        seek_frame = ttk.Frame(frame_player)
        seek_frame.pack(fill="x", padx=10, pady=(14, 0))

        time_row = ttk.Frame(seek_frame)
        time_row.pack(fill="x")
        self.time_current_label = ttk.Label(time_row, text="0:00", style="Muted.TLabel", background=COLOR_BG)
        self.time_current_label.pack(side="left")
        self.time_total_label = ttk.Label(time_row, text="0:00", style="Muted.TLabel", background=COLOR_BG)
        self.time_total_label.pack(side="right")

        self.seek_var = tk.DoubleVar(value=0.0)
        self.seek_scale = ttk.Scale(
            seek_frame, from_=0, to=100, orient="horizontal",
            variable=self.seek_var, style="Seek.Horizontal.TScale",
        )
        self.seek_scale.pack(fill="x", pady=(2, 0))
        self.seek_scale.bind("<ButtonPress-1>", self._on_seek_press)
        self.seek_scale.bind("<ButtonRelease-1>", self._on_seek_release)
        self.seek_scale.state(["disabled"])

        # -- Master volume --
        vol_frame = ttk.Frame(frame_player)
        vol_frame.pack(fill="x", padx=10, pady=(16, 0))
        ttk.Label(vol_frame, text="Master volume:").pack(side="left")
        self.vol_value_label = ttk.Label(vol_frame, text="70%")
        self.vol_value_label.pack(side="right")
        vol_scale = ttk.Scale(
            vol_frame, from_=0, to=100, orient="horizontal",
            variable=self.volume, command=self.on_master_volume_change
        )
        vol_scale.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(
            frame_player,
            text="Raise or lower each instrument to build your own mix, or silence it with \"Mute\".",
            style="Muted.TLabel", background=COLOR_BG
        ).pack(anchor="w", padx=10, pady=(10, 0))

        self.stems_frame = ttk.Frame(frame_player)
        self.stems_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.stem_rows = {}
        self._render_stem_placeholders(STEM_NAMES_6)

        # bottom breathing room so the scrollbar doesn't crop content
        ttk.Frame(root, height=16).pack(fill="x")

    def _render_stem_placeholders(self, stem_list):
        self.current_stem_list = stem_list
        for widget in self.stems_frame.winfo_children():
            widget.destroy()
        self.stem_rows = {}
        for stem in stem_list:
            row = tk.Frame(self.stems_frame, bg=COLOR_BG_ALT)
            row.pack(fill="x", pady=4)
            inner = tk.Frame(row, bg=COLOR_BG_ALT)
            inner.pack(fill="x", padx=10, pady=8)

            label = tk.Label(inner, text=STEM_LABELS[stem], width=9, anchor="w",
                              bg=COLOR_BG_ALT, fg=COLOR_TEXT, font=("Segoe UI Semibold", 9))
            label.pack(side="left")

            mute_var = tk.BooleanVar(value=False)
            mute_btn = ttk.Checkbutton(
                inner, text="Mute", variable=mute_var,
                command=lambda s=stem: self.on_mute_toggle(s),
                state="disabled",
            )
            mute_btn.pack(side="left", padx=(6, 10))

            stem_vol = tk.DoubleVar(value=100.0)
            vol_label = tk.Label(inner, text="100%", width=5, bg=COLOR_BG_ALT, fg=COLOR_TEXT_MUTED)
            scale = ttk.Scale(
                inner, from_=0, to=100, orient="horizontal",
                variable=stem_vol,
                command=lambda v, s=stem: self.on_stem_volume_change(s, v),
                state="disabled",
            )
            scale.pack(side="left", fill="x", expand=True, padx=5)
            vol_label.pack(side="left", padx=(0, 10))

            path_label = tk.Label(inner, text="(not generated yet)", bg=COLOR_BG_ALT, fg=COLOR_TEXT_MUTED)
            path_label.pack(side="left", padx=5)

            self.stem_rows[stem] = {
                "mute_var": mute_var,
                "mute_btn": mute_btn,
                "vol_var": stem_vol,
                "vol_label": vol_label,
                "scale": scale,
                "path_label": path_label,
                "sound": None,
                "channel": None,
            }

    # ---------------------------------------------------------- callbacks
    def choose_output_folder(self):
        folder = filedialog.askdirectory(title="Choose the parent folder")
        if folder:
            self.output_root.set(folder)

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Choose an audio file",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"), ("All files", "*.*")],
        )
        if file_path:
            self._set_input_file(file_path)

    def on_drop(self, event):
        raw = event.data
        path = raw.strip("{}")
        self._set_input_file(path)

    def _set_input_file(self, path):
        self.input_file.set(path)
        self.file_label.config(text=f"Selected file: {path}")
        self.status_text.set("Ready to separate. Choose a mode and press 'Separate instruments'.")

    def start_separation(self):
        input_path = self.input_file.get()
        output_root = self.output_root.get()
        mode = self.mode.get()

        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Please select a valid audio file first.")
            return
        if not output_root:
            messagebox.showerror("Error", "Please select an output parent folder.")
            return

        Path(output_root).mkdir(parents=True, exist_ok=True)

        stem_list = STEM_NAMES_4 if mode == MODE_ENSEMBLE else STEM_NAMES_6

        self.separate_btn.config(state="disabled")
        self.progress.start(10)
        self._render_stem_placeholders(stem_list)
        self.status_text.set("Separating instruments, this may take several minutes...")
        self._log_clear()
        self.stop_playback()

        thread = threading.Thread(
            target=run_separation,
            args=(input_path, output_root, mode, self.log_queue),
            daemon=True,
        )
        thread.start()

    def _poll_queue(self):
        try:
            while True:
                kind, payload = self.log_queue.get_nowait()
                if kind == "status":
                    self.status_text.set(payload)
                elif kind == "log":
                    self._log_append(payload)
                elif kind == "error":
                    self.progress.stop()
                    self.separate_btn.config(state="normal")
                    self.status_text.set(f"Error: {payload}")
                    messagebox.showerror("Error during separation", payload)
                elif kind == "done":
                    self.progress.stop()
                    self.separate_btn.config(state="normal")
                    self.status_text.set(
                        f"Done. Files saved to: {payload['folder']}"
                    )
                    self.current_stem_files = payload["files"]
                    self._update_stem_rows()
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _update_stem_rows(self):
        max_length = 0.0
        for stem, info in self.stem_rows.items():
            if stem in self.current_stem_files:
                path = self.current_stem_files[stem]
                try:
                    sound = pygame.mixer.Sound(path)
                    length = sound.get_length()
                    max_length = max(max_length, length)
                except Exception as e:
                    self._log_append(f"Could not load {stem}: {e}")
                    sound = None
                info["sound"] = sound
                info["channel"] = None
                info["mute_var"].set(False)
                info["vol_var"].set(100.0)
                info["vol_label"].config(text="100%")
                info["mute_btn"].config(state="normal" if sound else "disabled")
                info["scale"].config(state="normal" if sound else "disabled")
                info["path_label"].config(
                    text=Path(path).name, foreground=COLOR_SUCCESS
                )
            else:
                info["sound"] = None
                info["channel"] = None
                info["mute_btn"].config(state="disabled")
                info["scale"].config(state="disabled")
                info["path_label"].config(text="(not generated)", foreground=COLOR_TEXT_MUTED)

        any_sound = any(info["sound"] is not None for info in self.stem_rows.values())
        self.play_all_btn.config(state="normal" if any_sound else "disabled")

        self.track_length_sec = max_length
        self.playback_offset_sec = 0.0
        self.playback_started_at = None
        self.is_playing = False
        self.seek_var.set(0.0)
        self.time_total_label.config(text=self._format_time(max_length))
        self.time_current_label.config(text="0:00")
        if any_sound:
            self.seek_scale.state(["!disabled"])
        else:
            self.seek_scale.state(["disabled"])

    def _log_clear(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _log_append(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ---------------------------------------------------------- audio playback (mixer)
    def on_master_volume_change(self, value):
        vol = float(value)
        self.vol_value_label.config(text=f"{int(vol)}%")
        self._apply_all_volumes()

    def on_stem_volume_change(self, stem, value):
        vol = float(value)
        self.stem_rows[stem]["vol_label"].config(text=f"{int(vol)}%")
        self._apply_stem_volume(stem)

    def on_mute_toggle(self, stem):
        self._apply_stem_volume(stem)

    def _apply_stem_volume(self, stem):
        info = self.stem_rows.get(stem)
        if not info or info["channel"] is None:
            return
        if info["mute_var"].get():
            info["channel"].set_volume(0.0)
        else:
            master = self.volume.get() / 100.0
            stem_vol = info["vol_var"].get() / 100.0
            info["channel"].set_volume(max(0.0, min(1.0, master * stem_vol)))

    def _apply_all_volumes(self):
        for stem in self.stem_rows:
            self._apply_stem_volume(stem)

    def play_all(self):
        """Plays all available stems at the same time, each on its own
        pygame channel, so they can be mixed live using the sliders."""
        self._start_playback_from(self.playback_offset_sec if not self.is_playing else 0.0)

    def _start_playback_from(self, start_seconds):
        """(Re)starts playback of every loaded stem from start_seconds."""
        self._stop_channels_only()

        start_seconds = max(0.0, min(start_seconds, self.track_length_sec or 0.0))
        started_any = False
        for stem, info in self.stem_rows.items():
            sound = info["sound"]
            if sound is None:
                continue
            channel = sound.play(loops=0)
            if channel is None:
                self._log_append(f"Could not assign an audio channel for {stem} (too many active channels?)")
                continue
            try:
                channel.set_pos(start_seconds)
            except Exception:
                pass
            info["channel"] = channel
            started_any = True

        if not started_any:
            messagebox.showwarning("Notice", "No instruments have been generated yet to play.")
            return

        self._apply_all_volumes()
        self.playback_offset_sec = start_seconds
        import time as _time
        self.playback_started_at = _time.time()
        self.is_playing = True
        self.status_text.set("Playing mix. Adjust the sliders to blend the instruments, or drag the seek bar.")

    def _stop_channels_only(self):
        for info in self.stem_rows.values():
            if info["channel"] is not None:
                try:
                    info["channel"].stop()
                except Exception:
                    pass
                info["channel"] = None

    def stop_playback(self):
        self._stop_channels_only()
        self.is_playing = False
        self.playback_started_at = None
        self.status_text.set("Playback stopped.")

    # ---------------------------------------------------------- seek bar
    def _format_time(self, seconds):
        seconds = max(0, int(seconds or 0))
        minutes = seconds // 60
        secs = seconds % 60
        return f"{minutes}:{secs:02d}"

    def _current_playback_position(self):
        if not self.is_playing or self.playback_started_at is None:
            return self.playback_offset_sec
        import time as _time
        elapsed = _time.time() - self.playback_started_at
        pos = self.playback_offset_sec + elapsed
        return min(pos, self.track_length_sec or 0.0)

    def _on_seek_press(self, event):
        self.is_seeking = True

    def _on_seek_release(self, event):
        self.is_seeking = False
        if self.track_length_sec <= 0:
            return
        fraction = self.seek_var.get() / 100.0
        target_seconds = fraction * self.track_length_sec
        was_playing = self.is_playing
        if was_playing:
            self._start_playback_from(target_seconds)
        else:
            self.playback_offset_sec = target_seconds
            self.time_current_label.config(text=self._format_time(target_seconds))

    def _tick_seekbar(self):
        if self.track_length_sec > 0 and not self.is_seeking:
            pos = self._current_playback_position()
            if self.is_playing and pos >= self.track_length_sec:
                # Track finished playing
                self.stop_playback()
                pos = self.track_length_sec
                self.playback_offset_sec = 0.0
            fraction = 0.0
            if self.track_length_sec > 0:
                fraction = max(0.0, min(1.0, pos / self.track_length_sec))
            self.seek_var.set(fraction * 100.0)
            self.time_current_label.config(text=self._format_time(pos))
        self.root.after(200, self._tick_seekbar)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    app = SeparatorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()