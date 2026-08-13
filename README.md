# Separador de Instrumentos con IA (Demucs `htdemucs_6s`)

Aplicación de escritorio en Python que separa una canción en 6 instrumentos
usando el modelo `htdemucs_6s` de Demucs, con interfaz gráfica de
arrastrar y soltar, y un reproductor integrado con control de volumen.

## Sobre este proyecto
Desarrollado mediante vibecoding con Claude, si no te gusta esto no te juzgaria.

## Modos de separación (eliges uno por cada canción)

**Modo 1 — 6 instrumentos (rápido)**
Usa el modelo `htdemucs_6s`. Separa: voz, batería, bajo, guitarra, piano y otros
(cualquier instrumento que no entre en las categorías anteriores, ej. saxofón,
sintetizadores, cuerdas, etc.)

**Modo 2 — 4 instrumentos en Alta Calidad (ensemble)**
Corre **tres modelos distintos** (`htdemucs`, `htdemucs_ft`, `mdx_extra`) sobre
la misma canción y **promedia** el resultado de cada instrumento entre los tres.
Esto reduce artefactos y el efecto de sonido "ahogado"/"underwater" que a veces
se nota en canciones muy densas o muy comprimidas, porque los errores de cada
modelo tienden a no repetirse en los otros dos. A cambio:
- Tarda aproximadamente el triple (corre 3 modelos en vez de 1)
- Solo separa 4 instrumentos: voz, batería, bajo, otros — **no** separa
  guitarra ni piano por separado

**Modo 3 — 6 instrumentos en Alta Calidad (híbrido)**
Combina lo mejor de los dos modos anteriores, ya que no existe otro modelo
público que separe guitarra/piano para poder hacerles ensemble directo:
- **Voz, batería y bajo**: se obtienen igual que en el Modo 2, promediando
  los 3 modelos (`htdemucs`, `htdemucs_ft`, `mdx_extra`).
- **Guitarra, piano y otros**: se obtienen de `htdemucs_6s`, pero corrido
  con la técnica de **"shifts"** (la canción se procesa varias veces con
  pequeños desplazamientos de tiempo y se promedian los resultados, como
  un "ensemble contra sí mismo") y **overlap alto** entre fragmentos, lo
  que mejora la precisión sin perder la separación de guitarra/piano —
  ya que `htdemucs_6s` es el único modelo que sabe distinguirlos de
  "otros" en primer lugar.

Es el modo con mejor calidad para los 6 instrumentos completos, pero también
el más lento: corre 4 pasadas de Demucs en total (los 3 modelos del ensemble,
más `htdemucs_6s` con shifts, que de por sí tarda varias veces más que una
pasada normal por los desplazamientos internos).

> Nota: no existe un modelo público de Demucs que separe saxofón
> individualmente en ningún modo; siempre queda agrupado dentro de "Otros".

## Instalación

1. **Instala Python 3.10 o superior** (recomendado 3.10–3.12).

2. Crea un entorno virtual (opcional pero recomendado):

   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. Instala las dependencias:

   ```bash
   pip install -r requirements.txt
   ```

   La primera vez que ejecutes una separación, Demucs descargará
   automáticamente el modelo `htdemucs_6s` (~aprox 300-500 MB), así que
   necesitas conexión a internet la primera vez.

4. **FFmpeg**: Demucs lo necesita para leer/escribir MP3 y otros formatos
   comprimidos. Instálalo si no lo tienes:
   - Windows: descarga desde https://ffmpeg.org/download.html y agrégalo al PATH,
     o usa `winget install ffmpeg` / `choco install ffmpeg`.
   - macOS: `brew install ffmpeg`
   - Linux: `sudo apt install ffmpeg` (Debian/Ubuntu) o el equivalente de tu distro.

5. **Si tienes una GPU NVIDIA con CUDA (recomendado, especialmente para el
   modo Alta Calidad)**: instala la versión de PyTorch con soporte CUDA
   *antes* de instalar el resto de dependencias, para que Demucs la use
   automáticamente:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu121
   pip install -r requirements.txt
   ```

   Verifica que se detectó correctamente:

   ```bash
   python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
   ```

   Debería imprimir `True` y el nombre de tu tarjeta (ej. `NVIDIA GeForce RTX 4060`).
   El programa también muestra esto mismo en la parte superior de la ventana
   al abrirlo, para que sepas si está usando GPU o CPU sin tener que revisar
   la consola.

   Sin GPU, Demucs funciona igual mediante CPU, solo que más lento —el modo
   Alta Calidad en particular puede tardar bastante en CPU, ya que corre 3
   modelos en vez de 1.

## Uso

```bash
python separador.py
```

1. **Elige la carpeta madre**: es la carpeta donde se guardarán todas las
   separaciones. Cada canción crea su propia subcarpeta dentro de ella:
   `CarpetaMadre/Instrumentos NombreDeLaCancion (4 stems)/` o
   `CarpetaMadre/Instrumentos NombreDeLaCancion (6 stems)/` según el modo
   elegido.

2. **Arrastra tu canción** al recuadro (o haz clic para elegirla manualmente
   desde el explorador de archivos).

3. Presiona **"Separar instrumentos"**. Verás el progreso en el cuadro de
   estado/log. Este proceso puede tardar desde ~1 minuto (con GPU) hasta
   varios minutos (con CPU), dependiendo de la duración de la canción y tu
   hardware.

4. Cuando termine, se activa el **mezclador** en la parte inferior:
   - Presiona **"Reproducir mezcla"** y todos los instrumentos generados
     empiezan a sonar **al mismo tiempo, en loop continuo**.
   - Cada instrumento tiene su propio **slider de volumen** (0–100%) y un
     botón de **Mute** — puedes subir la guitarra, bajar la batería, o
     quitar completamente la voz mientras suena, todo en vivo.
   - El **volumen general (maestro)** escala el volumen de todos los
     instrumentos a la vez, además de sus volúmenes individuales.
   - Presiona **"Detener"** para parar todo.

   Nota: la reproducción es en loop (se repite) para que puedas seguir
   ajustando la mezcla sin que se corte a mitad de la canción.

5. Repite el proceso con otra canción: se creará automáticamente una nueva
   subcarpeta (`Instrumentos NombreDeOtraCancion (4/6 stems)`) dentro de la misma
   carpeta madre, sin sobrescribir la anterior.

## Estructura de carpetas resultante

```
CarpetaMadre/
├── Instrumentos CancionA (6 stems)/
│   ├── vocals.wav
│   ├── drums.wav
│   ├── bass.wav
│   ├── guitar.wav
│   ├── piano.wav
│   └── other.wav
└── Instrumentos CancionB (4 stems)/
    ├── vocals.wav
    ├── drums.wav
    ├── bass.wav
    └── other.wav
```

## Notas técnicas

- Si `tkinterdnd2` no se instala correctamente en tu sistema (a veces pasa
  en Linux dependiendo de la distribución), el programa sigue funcionando:
  simplemente haz clic en el recuadro para elegir el archivo con el
  explorador en vez de arrastrarlo.
- El programa ejecuta Demucs como un subproceso (`python -m demucs`), así
  que necesita que `demucs` esté instalado en el mismo entorno de Python
  desde el que corres `separador.py`.
- Los archivos de salida son `.wav` (sin compresión), por lo que pueden
  pesar varios cientos de MB por canción completa (los 6 stems juntos).

## Problemas comunes

- **"No se encontro 'demucs'"**: asegúrate de haber activado el entorno
  virtual correcto y de haber corrido `pip install -r requirements.txt`.
- **Error relacionado a ffmpeg / no puede leer el mp3**: instala ffmpeg
  como se indica arriba.
- **Muy lento**: es normal en CPU. Considera usar una GPU con CUDA, o
  procesar canciones más cortas primero para probar.
