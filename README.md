# AI Instrument Separator (Demucs `htdemucs_6s`)

A Python desktop application that separates a song into 6 instruments
using Demucs' `htdemucs_6s` model, with a drag-and-drop graphical
interface and a built-in mixer with per-instrument volume control.

## About this project
This project was built with the help of AI-assisted ("vibecoding") development.
The core logic is straightforward and has been tested, but as with any project
built this way, feel free to review the code before relying on it for anything
critical.

## Separation modes (choose one per song)

**Mode 1 — 6 instruments (fast)**
Uses the `htdemucs_6s` model. Separates: vocals, drums, bass, guitar, piano,
and other (any instrument that doesn't fall into the categories above, e.g.
saxophone, synths, strings, etc.)

**Mode 2 — 4 instruments in High Quality (ensemble)**
Runs **three different models** (`htdemucs`, `htdemucs_ft`, `mdx_extra`) on
the same song and **averages** the result for each instrument across all
three. This reduces artifacts and the "muffled"/"underwater" sound effect
sometimes noticeable in very dense or heavily compressed songs, since each
model's errors tend not to repeat across the other two. In exchange:
- It takes roughly three times as long (runs 3 models instead of 1)
- It only separates 4 instruments: vocals, drums, bass, other — it does
  **not** separate guitar or piano individually

**Mode 3 — 6 instruments in High Quality (hybrid)**
Combines the best of the two modes above, since there is no other public
model that separates guitar/piano so they could be ensembled directly:
- **Vocals, drums, and bass**: obtained the same way as in Mode 2, by
  averaging the 3 models (`htdemucs`, `htdemucs_ft`, `mdx_extra`).
- **Guitar, piano, and other**: obtained from `htdemucs_6s`, but run with
  the **"shifts"** technique (the song is processed several times with
  small time offsets and the results are averaged, like an "ensemble
  against itself") and a high **overlap** between chunks, which improves
  precision without losing the guitar/piano separation — since
  `htdemucs_6s` is the only model that can tell them apart from "other"
  in the first place.

This is the highest-quality mode for all 6 full instruments, but also the
slowest: it runs 4 Demucs passes in total (the 3 ensemble models, plus
`htdemucs_6s` with shifts, which by itself already takes several times
longer than a normal pass due to the internal offsets).

## Installation

1. **Install Python 3.10 or newer** (3.10–3.12 recommended).

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   The first time you run a separation, Demucs will automatically download
   the `htdemucs_6s` model (~300-500 MB), so you'll need an internet
   connection the first time.

4. **FFmpeg**: Demucs needs it to read/write MP3 and other compressed
   formats. Install it if you don't already have it:
   - Windows: download from https://ffmpeg.org/download.html and add it to
     your PATH, or use `winget install ffmpeg` / `choco install ffmpeg`.
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) or your distro's
     equivalent.

5. **If you have an NVIDIA GPU with CUDA (recommended, especially for High
   Quality mode)**: install the CUDA-enabled version of PyTorch *before*
   installing the rest of the dependencies, so Demucs picks it up
   automatically:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

   Verify it was detected correctly:

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

   It should print `True` and the name of your card (e.g.
   `NVIDIA GeForce RTX 4060`). The app also shows this same information at
   the top of the window when it opens, so you can tell whether it's using
   GPU or CPU without checking the console.

   Without a GPU, Demucs still works on CPU, just slower — High Quality
   mode in particular can take a while on CPU, since it runs 3 models
   instead of 1.

## Usage

```bash
python separator.py
```

1. **Choose the parent folder**: this is where all separations will be
   saved. Each song creates its own subfolder inside it:
   `ParentFolder/Instruments SongName (4 stems)/` or
   `ParentFolder/Instruments SongName (6 stems)/` depending on the chosen
   mode.

2. **Drag your song** into the drop zone (or click it to choose it
   manually from the file browser).

3. Press **"Separate instruments"**. You'll see progress in the status/log
   box. This process can take anywhere from ~1 minute (with a GPU) to
   several minutes (on CPU), depending on the song's length and your
   hardware.

4. When it finishes, the **mixer** at the bottom becomes active:
   - Press **"Play mix"** and all generated instruments start playing
     **at the same time**.
   - Each instrument has its own **volume slider** (0–100%) and a
     **Mute** button — you can raise the guitar, lower the drums, or
     completely remove the vocals while it's playing, all live.
   - The **master volume** scales the volume of every instrument at once,
     on top of their individual levels.
   - Use the **seek bar** to jump to any part of the song, just like on
     YouTube.
   - Press **"Stop"** to stop everything.

5. Repeat the process with another song: a new subfolder will be created
   automatically (`Instruments OtherSongName (4/6 stems)`) inside the same
   parent folder, without overwriting the previous one.

## Reopening previous separations

The app remembers the last parent folder you used, so the next time you
open it, that same folder is already selected. Right below it, a
**"Previous separations in this folder"** section lists every separation
it can find inside that folder (based on its folder name and the `.wav`
files still present), most recent first. For each one you can:

- **Load into mixer** — loads those stems straight into the mixer so you
  can play, seek, and blend them, without running Demucs again.
- **Open folder** — opens that separation's folder in your file explorer.
- **Refresh** — rescans the parent folder, useful if you moved or deleted
  folders manually.

This means you only need to run the separation process once per song —
after that, you can come back anytime and just reload the result.

## Resulting folder structure

```
ParentFolder/
├── Instruments SongA (6 stems)/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   ├── guitar.wav
│   ├── piano.wav
│   └── other.wav
└── Instruments SongB (4 stems)/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav
```

## Technical notes

- If `tkinterdnd2` doesn't install correctly on your system (this sometimes
  happens on Linux depending on the distribution), the app still works:
  just click the drop zone to choose the file through the file browser
  instead of dragging it.
- The app runs Demucs as a subprocess (`python -m demucs`), so `demucs`
  needs to be installed in the same Python environment you use to run
  `separator.py`.
- Output files are `.wav` (uncompressed), so a full set of stems can weigh
  several hundred MB per song.

## Common issues

- **"'demucs' was not found"**: make sure you've activated the correct
  virtual environment and run `pip install -r requirements.txt`.
- **ffmpeg-related error / can't read the mp3**: install ffmpeg as
  described above.
- **Very slow**: this is normal on CPU. Consider using a CUDA-enabled GPU,
  or try shorter songs first to test things out.