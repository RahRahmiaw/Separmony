"""
Separador de Instrumentos con IA (Demucs)
=========================================================
Aplicacion de escritorio con interfaz grafica que permite:
- Arrastrar y soltar un archivo de audio (o seleccionarlo manualmente)
- Elegir una carpeta "madre" donde se guardaran todas las separaciones
- Elegir por cancion entre tres modos:
    * 6 instrumentos (voz, bateria, bajo, guitarra, piano, otros)
      usando htdemucs_6s — rapido
    * 4 instrumentos en Alta Calidad (voz, bateria, bajo, otros)
      usando un ENSEMBLE (promedio) de htdemucs + htdemucs_ft + mdx_extra,
      lo cual reduce artefactos/sonido "ahogado" a costa de mas tiempo
      de procesamiento y sin separar guitarra/piano
    * 6 instrumentos en Alta Calidad (hibrido): voz/bateria/bajo salen
      del ensemble de 3 modelos (igual que el modo anterior), y
      guitarra/piano/otros salen de htdemucs_6s corrido con shifts y
      overlap altos para mayor precision — el modo mas lento pero el
      mas limpio disponible para los 6 stems completos
- Guardar los resultados en una subcarpeta "Instrumentos <NombreCancion> (4/6 stems)"
  dentro de la carpeta madre elegida
- Mezclador integrado: reproducir todos los instrumentos AL MISMO TIEMPO,
  con un slider de volumen independiente por cada uno (y boton de Mute),
  ademas de un volumen maestro general, para poder armar tu propia mezcla
  o aislar/quitar instrumentos en vivo
- Detectar y mostrar si se esta usando GPU (CUDA) o CPU

Requisitos (instalar antes de correr, ver requirements.txt):
    pip install -r requirements.txt

Uso:
    python separador.py
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
# Configuracion general
# ----------------------------------------------------------------------

MODEL_6STEM = "htdemucs_6s"
ENSEMBLE_MODELS = ["htdemucs", "htdemucs_ft", "mdx_extra"]  # se promedian entre si

STEM_NAMES_6 = ["vocals", "drums", "bass", "guitar", "piano", "other"]
STEM_NAMES_4 = ["vocals", "drums", "bass", "other"]

STEM_LABELS_ES = {
    "vocals": "Voz",
    "drums": "Bateria",
    "bass": "Bajo",
    "guitar": "Guitarra",
    "piano": "Piano",
    "other": "Otros",
}

APP_TITLE = "Separador de Instrumentos IA (Demucs)"

MODE_6STEM = "6stem"
MODE_ENSEMBLE = "ensemble"
MODE_HYBRID_6STEM = "hybrid6"

# Parametros de "shifts" y "overlap" para mejorar la precision de htdemucs_6s
# en el modo hibrido (mas lento, pero mas limpio para guitarra/piano/otros)
HYBRID_SHIFTS = 5
HYBRID_OVERLAP = 0.5


# ----------------------------------------------------------------------
# Utilidades de deteccion de GPU
# ----------------------------------------------------------------------

def get_device_info():
    if not TORCH_AVAILABLE:
        return "PyTorch no instalado (no se puede verificar GPU)."
    try:
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            return f"GPU detectada: {name} (se usara CUDA automaticamente)"
        else:
            return "No se detecto GPU compatible con CUDA. Se usara CPU (mas lento)."
    except Exception as e:
        return f"No se pudo verificar el dispositivo: {e}"


# ----------------------------------------------------------------------
# Logica de separacion (corre en un hilo aparte para no congelar la UI)
# ----------------------------------------------------------------------

def _make_safe_copy(input_file, temp_dir):
    """
    Copia el archivo de entrada a un nombre de archivo "seguro" dentro de
    temp_dir, evitando caracteres problematicos en Windows (puntos finales,
    signos de exclamacion, comillas, etc. suelen causar que Windows trunque
    o rechace la carpeta que demucs intenta crear con ese mismo nombre).
    Devuelve la ruta al archivo copiado.
    """
    ext = Path(input_file).suffix
    safe_name = f"input_audio{ext}"
    safe_path = Path(temp_dir) / safe_name
    shutil.copy(str(input_file), str(safe_path))
    return safe_path


def _run_single_demucs(model_name, input_file, work_dir, log_queue, extra_args=None):
    """Corre demucs para un solo modelo y devuelve la carpeta con los wavs."""
    cmd = [
        sys.executable, "-m", "demucs",
        "-n", model_name,
        "-o", str(work_dir),
    ]
    if extra_args:
        cmd.extend(extra_args)
    cmd.append(str(input_file))

    log_queue.put(("log", f"--- Ejecutando modelo: {model_name} {' '.join(extra_args or [])} ---"))
    process = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
    )
    for line in process.stdout:
        log_queue.put(("log", line.rstrip()))
    process.wait()
    if process.returncode != 0:
        raise RuntimeError(f"Demucs fallo con el modelo {model_name} (codigo {process.returncode})")

    song_name = Path(input_file).stem
    out_dir = Path(work_dir) / model_name / song_name
    if not out_dir.exists():
        raise RuntimeError(f"No se genero la carpeta esperada para {model_name}: {out_dir}")
    return out_dir


def _average_wavs(paths, destination):
    """Promedia una lista de archivos .wav (misma duracion esperada) y guarda el resultado."""
    if not SOUNDFILE_AVAILABLE:
        raise RuntimeError("Falta la libreria 'soundfile' para promediar audio (pip install soundfile).")

    data_list = []
    samplerate = None
    for p in paths:
        data, sr = sf.read(str(p))
        if samplerate is None:
            samplerate = sr
        elif sr != samplerate:
            raise RuntimeError(f"Sample rates distintos entre modelos: {sr} vs {samplerate}")
        data_list.append(data)

    min_len = min(d.shape[0] for d in data_list)
    data_list = [d[:min_len] for d in data_list]

    averaged = np.mean(np.stack(data_list, axis=0), axis=0)
    sf.write(str(destination), averaged, samplerate)


def run_separation(input_file: str, output_root: str, mode: str, log_queue: "queue.Queue"):
    """
    Ejecuta la separacion segun el modo elegido:
    - MODE_6STEM: un solo modelo (htdemucs_6s), 6 stems.
    - MODE_ENSEMBLE: corre 3 modelos de 4 stems y promedia los resultados
      instrumento por instrumento para reducir artefactos.
    - MODE_HYBRID_6STEM: 6 stems combinando lo mejor de ambos mundos:
        * voz/bateria/bajo -> promedio (ensemble) de los 3 modelos de 4 stems
        * guitarra/piano/otros -> htdemucs_6s corrido con shifts+overlap
          altos (mas preciso), ya que es el unico modelo que sabe separar
          guitarra y piano de "otros" de forma confiable
    """
    song_name = Path(input_file).stem
    stems_label = {
        MODE_6STEM: "6 stems",
        MODE_ENSEMBLE: "4 stems",
        MODE_HYBRID_6STEM: "6 stems HQ",
    }.get(mode, "stems")
    final_folder = Path(output_root) / f"Instrumentos {song_name} ({stems_label})"
    final_folder.mkdir(parents=True, exist_ok=True)

    try:
        with tempfile.TemporaryDirectory(prefix="demucs_input_") as safe_input_dir:
            safe_input = _make_safe_copy(input_file, safe_input_dir)
            log_queue.put(("log", f"Copia segura del archivo de entrada: {safe_input.name}"))

            if mode == MODE_6STEM:
                log_queue.put(("status", f"Separando '{song_name}' en 6 instrumentos ({MODEL_6STEM})..."))
                out_dir = _run_single_demucs(MODEL_6STEM, safe_input, output_root, log_queue)

                moved_files = {}
                for stem in STEM_NAMES_6:
                    src = out_dir / f"{stem}.wav"
                    if src.exists():
                        dst = final_folder / f"{stem}.wav"
                        shutil.move(str(src), str(dst))
                        moved_files[stem] = str(dst)
                    else:
                        log_queue.put(("log", f"Aviso: no se genero {stem}.wav"))

                _cleanup_model_dir(output_root, MODEL_6STEM, out_dir)
                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            elif mode == MODE_ENSEMBLE:
                log_queue.put((
                    "status",
                    f"Separando '{song_name}' en Alta Calidad (ensemble de {len(ENSEMBLE_MODELS)} modelos)... "
                    "Esto tardara mas que el modo normal."
                ))

                with tempfile.TemporaryDirectory(prefix="ensemble_work_") as tmp_root:
                    model_out_dirs = []
                    for model_name in ENSEMBLE_MODELS:
                        out_dir = _run_single_demucs(model_name, safe_input, tmp_root, log_queue)
                        model_out_dirs.append(out_dir)

                    log_queue.put(("status", "Promediando resultados de los modelos para reducir artefactos..."))

                    moved_files = {}
                    for stem in STEM_NAMES_4:
                        stem_paths = [d / f"{stem}.wav" for d in model_out_dirs]
                        stem_paths = [p for p in stem_paths if p.exists()]
                        if not stem_paths:
                            log_queue.put(("log", f"Aviso: ningun modelo genero {stem}.wav"))
                            continue
                        dst = final_folder / f"{stem}.wav"
                        if len(stem_paths) == 1:
                            shutil.copy(str(stem_paths[0]), str(dst))
                        else:
                            _average_wavs(stem_paths, dst)
                        moved_files[stem] = str(dst)
                        log_queue.put(("log", f"{stem}: promediado de {len(stem_paths)} modelos -> {dst.name}"))

                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            elif mode == MODE_HYBRID_6STEM:
                log_queue.put((
                    "status",
                    f"Separando '{song_name}' en 6 instrumentos Alta Calidad "
                    f"(voz/bateria/bajo por ensemble + guitarra/piano/otros por {MODEL_6STEM} con shifts)... "
                    "Este es el modo mas lento."
                ))

                moved_files = {}

                # --- Parte 1: voz/bateria/bajo por ensemble de 3 modelos de 4 stems ---
                with tempfile.TemporaryDirectory(prefix="hybrid_ensemble_") as tmp_root:
                    model_out_dirs = []
                    for model_name in ENSEMBLE_MODELS:
                        out_dir = _run_single_demucs(model_name, safe_input, tmp_root, log_queue)
                        model_out_dirs.append(out_dir)

                    log_queue.put(("status", "Promediando voz/bateria/bajo entre los 3 modelos..."))
                    for stem in ["vocals", "drums", "bass"]:
                        stem_paths = [d / f"{stem}.wav" for d in model_out_dirs]
                        stem_paths = [p for p in stem_paths if p.exists()]
                        if not stem_paths:
                            log_queue.put(("log", f"Aviso: ningun modelo genero {stem}.wav"))
                            continue
                        dst = final_folder / f"{stem}.wav"
                        if len(stem_paths) == 1:
                            shutil.copy(str(stem_paths[0]), str(dst))
                        else:
                            _average_wavs(stem_paths, dst)
                        moved_files[stem] = str(dst)
                        log_queue.put(("log", f"{stem}: promediado de {len(stem_paths)} modelos -> {dst.name}"))

                # --- Parte 2: guitarra/piano/otros por htdemucs_6s con shifts+overlap ---
                with tempfile.TemporaryDirectory(prefix="hybrid_6s_") as tmp_root2:
                    log_queue.put((
                        "status",
                        f"Separando guitarra/piano/otros con {MODEL_6STEM} "
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
                            log_queue.put(("log", f"Aviso: no se genero {stem}.wav"))

                log_queue.put(("done", {"folder": str(final_folder), "files": moved_files, "song": song_name}))

            else:
                log_queue.put(("error", f"Modo desconocido: {mode}"))

    except FileNotFoundError:
        log_queue.put(("error", "No se encontro 'demucs'. Instalalo con: pip install demucs"))
    except Exception as e:
        log_queue.put(("error", f"Error inesperado: {e}"))


def _cleanup_model_dir(output_root, model_name, out_dir):
    try:
        shutil.rmtree(out_dir)
        model_dir = Path(output_root) / model_name
        if model_dir.exists() and not any(model_dir.iterdir()):
            model_dir.rmdir()
    except Exception:
        pass


# ----------------------------------------------------------------------
# Interfaz grafica
# ----------------------------------------------------------------------

class SeparadorApp:
    def __init__(self, root):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.geometry("740x700")
        self.root.minsize(700, 660)

        self.output_root = tk.StringVar(value=str(Path.home() / "Separaciones_IA"))
        self.input_file = tk.StringVar(value="")
        self.status_text = tk.StringVar(value="Esperando archivo de audio...")
        self.volume = tk.DoubleVar(value=70.0)
        self.mode = tk.StringVar(value=MODE_6STEM)

        self.log_queue = queue.Queue()
        self.current_stem_files = {}
        self.current_stem_list = STEM_NAMES_6

        pygame.mixer.init()
        pygame.mixer.set_num_channels(16)  # suficientes canales para los 6 stems simultaneos

        self._build_ui()
        self._poll_queue()

    # ---------------------------------------------------------- UI build
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        device_frame = ttk.Frame(self.root)
        device_frame.pack(fill="x", padx=10, pady=(8, 0))
        ttk.Label(device_frame, text=get_device_info(), foreground="#2a7f2a").pack(anchor="w")

        frame_out = ttk.LabelFrame(self.root, text="1. Carpeta madre donde se guardaran las separaciones")
        frame_out.pack(fill="x", **pad)

        entry_out = ttk.Entry(frame_out, textvariable=self.output_root)
        entry_out.pack(side="left", fill="x", expand=True, padx=(10, 5), pady=8)
        ttk.Button(frame_out, text="Elegir carpeta...", command=self.choose_output_folder).pack(
            side="left", padx=(0, 10), pady=8
        )

        frame_drop = ttk.LabelFrame(self.root, text="2. Arrastra tu cancion aqui (o haz clic para elegirla)")
        frame_drop.pack(fill="x", **pad)

        self.drop_label = tk.Label(
            frame_drop,
            text="Suelta el archivo de audio aqui\n(MP3, WAV, FLAC, M4A...)",
            bg="#2b2b2b", fg="white",
            height=4,
            font=("Segoe UI", 11),
            relief="groove",
        )
        self.drop_label.pack(fill="x", padx=10, pady=10)
        self.drop_label.bind("<Button-1>", lambda e: self.choose_input_file())

        if DND_AVAILABLE:
            self.drop_label.drop_target_register(DND_FILES)
            self.drop_label.dnd_bind("<<Drop>>", self.on_drop)
        else:
            self.drop_label.config(
                text="tkinterdnd2 no instalado: haz clic aqui para elegir el archivo\n"
                     "(instala tkinterdnd2 para poder arrastrar y soltar)"
            )

        self.file_label = ttk.Label(frame_drop, text="Ningun archivo seleccionado.")
        self.file_label.pack(padx=10, pady=(0, 10), anchor="w")

        frame_mode = ttk.LabelFrame(self.root, text="3. Modo de separacion para esta cancion")
        frame_mode.pack(fill="x", **pad)

        ttk.Radiobutton(
            frame_mode,
            text="6 instrumentos (voz, bateria, bajo, guitarra, piano, otros) — mas rapido",
            variable=self.mode, value=MODE_6STEM,
        ).pack(anchor="w", padx=10, pady=(8, 2))

        ttk.Radiobutton(
            frame_mode,
            text="4 instrumentos en Alta Calidad (voz, bateria, bajo, otros) — ensemble de 3 modelos,\n"
                 "reduce sonido \"ahogado\"/artefactos pero es mas lento y no separa guitarra ni piano",
            variable=self.mode, value=MODE_ENSEMBLE,
        ).pack(anchor="w", padx=10, pady=(2, 8))

        ttk.Radiobutton(
            frame_mode,
            text="6 instrumentos en Alta Calidad (voz/bateria/bajo por ensemble + guitarra/piano/otros\n"
                 "con shifts) — el mas limpio para los 6 stems, pero tambien el mas lento",
            variable=self.mode, value=MODE_HYBRID_6STEM,
        ).pack(anchor="w", padx=10, pady=(2, 8))

        frame_action = ttk.Frame(self.root)
        frame_action.pack(fill="x", **pad)
        self.separate_btn = ttk.Button(
            frame_action, text="Separar instrumentos", command=self.start_separation
        )
        self.separate_btn.pack(side="left")

        self.progress = ttk.Progressbar(frame_action, mode="indeterminate")
        self.progress.pack(side="left", fill="x", expand=True, padx=10)

        frame_status = ttk.LabelFrame(self.root, text="Estado")
        frame_status.pack(fill="x", **pad)
        ttk.Label(frame_status, textvariable=self.status_text, wraplength=700).pack(
            anchor="w", padx=10, pady=5
        )

        self.log_box = tk.Text(frame_status, height=6, state="disabled", bg="#1e1e1e", fg="#cfcfcf")
        self.log_box.pack(fill="both", padx=10, pady=(0, 10))

        frame_player = ttk.LabelFrame(self.root, text="4. Mezclador: escucha y combina los instrumentos separados")
        frame_player.pack(fill="both", expand=True, **pad)

        transport_frame = ttk.Frame(frame_player)
        transport_frame.pack(fill="x", padx=10, pady=(10, 0))
        self.play_all_btn = ttk.Button(
            transport_frame, text="Reproducir mezcla", state="disabled", command=self.play_all
        )
        self.play_all_btn.pack(side="left")
        ttk.Button(transport_frame, text="Detener", command=self.stop_playback).pack(
            side="left", padx=8
        )

        vol_frame = ttk.Frame(frame_player)
        vol_frame.pack(fill="x", padx=10, pady=(10, 0))
        ttk.Label(vol_frame, text="Volumen general (maestro):").pack(side="left")
        self.vol_value_label = ttk.Label(vol_frame, text="70%")
        self.vol_value_label.pack(side="right")
        vol_scale = ttk.Scale(
            vol_frame, from_=0, to=100, orient="horizontal",
            variable=self.volume, command=self.on_master_volume_change
        )
        vol_scale.pack(side="left", fill="x", expand=True, padx=10)

        ttk.Label(
            frame_player,
            text="Sube o baja cada instrumento para armar tu propia mezcla, o silencialo con \"Mute\".",
            foreground="gray"
        ).pack(anchor="w", padx=10, pady=(6, 0))

        self.stems_frame = ttk.Frame(frame_player)
        self.stems_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.stem_rows = {}
        self._render_stem_placeholders(STEM_NAMES_6)

    def _render_stem_placeholders(self, stem_list):
        self.current_stem_list = stem_list
        for widget in self.stems_frame.winfo_children():
            widget.destroy()
        self.stem_rows = {}
        for stem in stem_list:
            row = ttk.Frame(self.stems_frame)
            row.pack(fill="x", pady=4)

            label = ttk.Label(row, text=STEM_LABELS_ES[stem], width=10)
            label.pack(side="left")

            mute_var = tk.BooleanVar(value=False)
            mute_btn = ttk.Checkbutton(
                row, text="Mute", variable=mute_var,
                command=lambda s=stem: self.on_mute_toggle(s),
                state="disabled",
            )
            mute_btn.pack(side="left", padx=(0, 8))

            stem_vol = tk.DoubleVar(value=100.0)
            vol_label = ttk.Label(row, text="100%", width=5)
            scale = ttk.Scale(
                row, from_=0, to=100, orient="horizontal",
                variable=stem_vol,
                command=lambda v, s=stem: self.on_stem_volume_change(s, v),
                state="disabled",
            )
            scale.pack(side="left", fill="x", expand=True, padx=5)
            vol_label.pack(side="left", padx=(0, 10))

            path_label = ttk.Label(row, text="(no generado aun)", foreground="gray")
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
        folder = filedialog.askdirectory(title="Elige la carpeta madre")
        if folder:
            self.output_root.set(folder)

    def choose_input_file(self):
        file_path = filedialog.askopenfilename(
            title="Elige un archivo de audio",
            filetypes=[("Audio", "*.mp3 *.wav *.flac *.m4a *.ogg *.aac"), ("Todos", "*.*")],
        )
        if file_path:
            self._set_input_file(file_path)

    def on_drop(self, event):
        raw = event.data
        path = raw.strip("{}")
        self._set_input_file(path)

    def _set_input_file(self, path):
        self.input_file.set(path)
        self.file_label.config(text=f"Archivo seleccionado: {path}")
        self.status_text.set("Listo para separar. Elige el modo y presiona 'Separar instrumentos'.")

    def start_separation(self):
        input_path = self.input_file.get()
        output_root = self.output_root.get()
        mode = self.mode.get()

        if not input_path or not os.path.isfile(input_path):
            messagebox.showerror("Error", "Selecciona primero un archivo de audio valido.")
            return
        if not output_root:
            messagebox.showerror("Error", "Selecciona una carpeta madre de salida.")
            return

        Path(output_root).mkdir(parents=True, exist_ok=True)

        stem_list = STEM_NAMES_4 if mode == MODE_ENSEMBLE else STEM_NAMES_6

        self.separate_btn.config(state="disabled")
        self.progress.start(10)
        self._render_stem_placeholders(stem_list)
        self.status_text.set("Separando instrumentos, esto puede tardar varios minutos...")
        self._log_clear()

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
                    messagebox.showerror("Error durante la separacion", payload)
                elif kind == "done":
                    self.progress.stop()
                    self.separate_btn.config(state="normal")
                    self.status_text.set(
                        f"Listo. Archivos guardados en: {payload['folder']}"
                    )
                    self.current_stem_files = payload["files"]
                    self._update_stem_rows()
        except queue.Empty:
            pass
        self.root.after(150, self._poll_queue)

    def _update_stem_rows(self):
        for stem, info in self.stem_rows.items():
            if stem in self.current_stem_files:
                path = self.current_stem_files[stem]
                try:
                    sound = pygame.mixer.Sound(path)
                except Exception as e:
                    self._log_append(f"No se pudo cargar {stem}: {e}")
                    sound = None
                info["sound"] = sound
                info["channel"] = None
                info["mute_var"].set(False)
                info["vol_var"].set(100.0)
                info["vol_label"].config(text="100%")
                info["mute_btn"].config(state="normal" if sound else "disabled")
                info["scale"].config(state="normal" if sound else "disabled")
                info["path_label"].config(
                    text=Path(path).name, foreground="green"
                )
            else:
                info["sound"] = None
                info["channel"] = None
                info["mute_btn"].config(state="disabled")
                info["scale"].config(state="disabled")
                info["path_label"].config(text="(no generado)", foreground="gray")

        any_sound = any(info["sound"] is not None for info in self.stem_rows.values())
        self.play_all_btn.config(state="normal" if any_sound else "disabled")

    def _log_clear(self):
        self.log_box.config(state="normal")
        self.log_box.delete("1.0", "end")
        self.log_box.config(state="disabled")

    def _log_append(self, text):
        self.log_box.config(state="normal")
        self.log_box.insert("end", text + "\n")
        self.log_box.see("end")
        self.log_box.config(state="disabled")

    # ---------------------------------------------------------- audio playback (mezclador)
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
        """Reproduce todos los stems disponibles al mismo tiempo, cada uno en su
        propio canal de pygame, para poder mezclarlos en vivo con los sliders."""
        self.stop_playback()

        started_any = False
        for stem, info in self.stem_rows.items():
            sound = info["sound"]
            if sound is None:
                continue
            channel = sound.play(loops=-1)  # loop continuo para poder mezclar comodamente
            if channel is None:
                self._log_append(f"No se pudo asignar canal de audio para {stem} (¿demasiados canales activos?)")
                continue
            info["channel"] = channel
            started_any = True

        if not started_any:
            messagebox.showwarning("Aviso", "No hay instrumentos generados todavia para reproducir.")
            return

        self._apply_all_volumes()
        self.status_text.set("Reproduciendo mezcla. Ajusta los sliders para combinar los instrumentos.")

    def stop_playback(self):
        for info in self.stem_rows.values():
            if info["channel"] is not None:
                try:
                    info["channel"].stop()
                except Exception:
                    pass
                info["channel"] = None
        self.status_text.set("Reproduccion detenida.")


# ----------------------------------------------------------------------
# Punto de entrada
# ----------------------------------------------------------------------

def main():
    if DND_AVAILABLE:
        root = TkinterDnD.Tk()
    else:
        root = tk.Tk()

    style = ttk.Style(root)
    try:
        style.theme_use("clam")
    except Exception:
        pass

    app = SeparadorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
