# Separmony: AI Instrument Separator (Demucs + BS-Roformer-SW)

A Python desktop application that separates a song into its instruments
using Demucs and BS-Roformer-SW, with a drag-and-drop graphical interface
and a built-in mixer with per-instrument volume control.

## About this project
This project was built with the help of AI-assisted development.
The core logic is straightforward and has been tested, but as with any project
built this way, feel free to review the code before relying on it for anything
critical.

## Separation modes (choose one per song)

**Mode 1: 6 instruments (fast)**
Uses the `htdemucs_6s` model. Separates: vocals, drums, bass, guitar, piano,
and other (any instrument that doesn't fall into the categories above, e.g.
saxophone, synths, strings, etc.). Guitar and piano are the weakest part of
this model and can sound artifacted.

**Mode 2: 4 instruments in High Quality (ensemble)**
Runs **three different models** (`htdemucs`, `htdemucs_ft`, `mdx_extra`) on
the same song and **averages** the result for each instrument across all
three. This reduces artifacts and the "muffled"/"underwater" sound effect
sometimes noticeable in very dense or heavily compressed songs, since each
model's errors tend not to repeat across the other two. In exchange:
- It takes roughly three times as long (runs 3 models instead of 1)
- It only separates 4 instruments: vocals, drums, bass, other. It does
  **not** separate guitar or piano individually

**Mode 3: 6 instruments in High Quality (hybrid)**
Combines the ensemble with a much stronger guitar/piano model:
- **Vocals, drums, and bass**: obtained the same way as in Mode 2, by
  averaging the 3 Demucs models (`htdemucs`, `htdemucs_ft`, `mdx_extra`).
- **Guitar and piano**: obtained from **BS-Roformer-SW**, a 6-stem Band-Split
  RoFormer model run through the `audio-separator` package. It is clearly
  cleaner for guitar and piano than `htdemucs_6s`. By default it processes
  the **original song**.
- **Other**: the ensemble's "other" stem **minus** guitar and piano, so the
  six stems together stay close to the original mix without duplicating
  instruments.

It is the highest-quality mode for all 6 instruments, but also the slowest:
3 Demucs passes plus 1 BS-Roformer-SW pass.

> **Advanced:** at the top of `separmony.py` there is a constant
> `STEP2_SOURCE`. With `"full"` (default) BS-Roformer-SW receives the
> original song. With `"other"` it receives only the ensemble's averaged
> "other" stem, and the final "other" is built by summing BS-Roformer-SW's
> leftover stems. You can try both and keep whichever sounds better.

## Installation

1. **Install Python 3.10 or newer** (3.10-3.12 recommended).

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **If you have an NVIDIA GPU with CUDA (recommended, especially for the
   High Quality modes)**: install the CUDA-enabled version of PyTorch
   *first*:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install the dependencies:

   ```bash
   pip install -r requirements.txt
   ```

   Without a GPU, edit `requirements.txt` and change `audio-separator[gpu]`
   to `audio-separator[cpu]` before running the command.

5. **Check that your GPU is still detected.** Installing `audio-separator`
   can replace a CUDA build of PyTorch with a CPU-only one. Verify:

   ```bash
   python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
   ```

   It should print a version ending in `+cu121` (or similar) and `True`. If
   it prints `False`, reinstall the CUDA build from step 3. The app also
   shows this information at the top of the window when it opens.

6. **FFmpeg**: required to read/write MP3 and other compressed formats, and
   required by `audio-separator` (Mode 3). Install it if you don't already
   have it:
   - Windows: `winget install Gyan.FFmpeg`
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) or your distro's
     equivalent.

   On Windows the app looks for FFmpeg automatically when it starts (it
   reloads PATH from the registry and searches winget's install folders),
   so you don't need to restart your terminal after installing it.

### Model downloads (first run only)

Models are downloaded automatically the first time they are needed, so you
need an internet connection:
- Demucs models are downloaded by Demucs (a few hundred MB each).
- BS-Roformer-SW (Mode 3) is about 700 MB and is stored in
  `~/.ai_instrument_separator/models`.

## Usage

```bash
python separmony.py
```

1. **Choose the parent folder**: this is where all separations will be
   saved. Each song creates its own subfolder inside it, named after the
   mode:
   - `Instruments SongName (6 stems)` (Mode 1)
   - `Instruments SongName (4 stems)` (Mode 2)
   - `Instruments SongName (6 stems HQ)` (Mode 3)

2. **Drag your song** into the drop zone (or click it to choose it
   manually from the file browser).

3. Press **"Separate instruments"**. You'll see progress in the status/log
   box. This can take from a couple of minutes (Mode 1 with a GPU) to a
   long time (Mode 3, or any mode on CPU), depending on the song's length
   and your hardware. Try a short song first.

4. When it finishes, the **mixer** at the bottom becomes active:
   - Press **"Play mix"** and all generated instruments start playing
     **at the same time**.
   - Each instrument has its own **volume slider** (0-100%) and a
     **Mute** button. You can raise the guitar, lower the drums, or
     completely remove the vocals while it's playing, all live.
   - The **master volume** scales the volume of every instrument at once,
     on top of their individual levels.
   - Use the **seek bar** to jump to any part of the song, just like on
     YouTube.
   - Press **"Stop"** to stop everything.

5. Repeat the process with another song: a new subfolder will be created
   automatically inside the same parent folder, without overwriting the
   previous one.

## Reopening previous separations

The app remembers the last parent folder you used, so the next time you
open it, that same folder is already selected. Right below it, a
**"Previous separations in this folder"** section lists every separation
it can find inside that folder (based on its folder name and the `.wav`
files still present), most recent first. For each one you can:

- **Load into mixer**: loads those stems straight into the mixer so you
  can play, seek, and blend them, without running the separation again.
- **Open folder**: opens that separation's folder in your file explorer.
- **Refresh**: rescans the parent folder, useful if you moved or deleted
  folders manually.

This means you only need to run the separation process once per song;
after that, you can come back anytime and just reload the result.

## Resulting folder structure

```
ParentFolder/
├── Instruments SongA (6 stems HQ)/
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
- Demucs runs as a subprocess (`python -m demucs`), so `demucs` needs to be
  installed in the same Python environment you use to run `separmony.py`.
  BS-Roformer-SW runs inside the app's own process through the
  `audio-separator` Python package.
- Output files are `.wav` (uncompressed), so a full set of stems can weigh
  several hundred MB per song.
- Settings (last parent folder) are stored in
  `~/.ai_instrument_separator/settings.json`.

## Common issues

- **"A required program or file was not found"**: this usually means
  `demucs` or `ffmpeg` can't be found. Make sure you've activated the
  correct virtual environment, ran `pip install -r requirements.txt`, and
  installed FFmpeg as described above.
- **"Missing 'audio-separator'"**: Mode 3 needs it. Run
  `pip install "audio-separator[gpu]"` (or `[cpu]`).
- **The app says it's using the CPU after installing audio-separator**:
  pip replaced your CUDA PyTorch with a CPU build. Reinstall it with
  `pip install torch --index-url https://download.pytorch.org/whl/cu121`
  and check again with the command from the Installation section.
- **Very slow**: this is normal on CPU, and Mode 3 is the slowest mode.
  Consider a CUDA-enabled GPU, or try shorter songs first.