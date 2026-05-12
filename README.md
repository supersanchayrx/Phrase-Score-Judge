# Instructions — Running the Pitch Scoring Pipeline

## Dependencies

Install the following Python packages before running the pipeline:

```bash
pip install numpy librosa soundfile scipy matplotlib swift-f0 audio-separator onnxruntime
```

| Package            | Purpose                                                  |
|--------------------|----------------------------------------------------------|
| `numpy`            | Numerical array operations                               |
| `librosa`          | Audio loading, resampling, and DTW computation           |
| `soundfile`        | Audio file I/O (WAV read/write)                          |
| `scipy`            | Signal processing utilities (FFT, median filter)         |
| `matplotlib`       | Pitch visualization plots                                |
| `swift-f0`         | Pitch detection, F0 extraction, VAD, and confidence      |
| `audio-separator`  | Neural vocal/instrumental source separation              |
| `onnxruntime`      | ONNX model inference backend (used by `audio-separator`) |

> **Note:** The vocal separation model (`Kim_Vocal_2.onnx`) will be automatically downloaded to the `models/` directory on the first run by `audio-separator`.

---

## Directory Structure

Before running, ensure your project directory is set up as follows:

```
project-root/
├── assets/
│   ├── reference/
│   │   ├── vocals.mp3          # Reference vocal track
│   │   └── struct.json         # Phrase timing metadata
│   └── tracks/
│       ├── candidate1.wav      # Candidate vocal tracks
│       ├── candidate2.wav
│       └── ...
├── main.py
├── PreProcessRefAudio.py
├── PreProcessCandidates.py
├── ProcessAudioArray.py
├── VocalsExtractor.py
├── dtw.py
└── models/                     # Created automatically (ONNX model cache)
```

### Required paths

| Path                        | Contents                                                       |
|-----------------------------|----------------------------------------------------------------|
| `assets/reference/`        | Reference vocal track (`vocals.mp3`) and phrase metadata (`struct.json`) |
| `assets/tracks/`           | All candidate `.wav` files to be scored                        |
| `assets/cleanAudioCache/`  | Auto-generated — cached vocal/instrumental stems from source separation |
| `models/`                  | Auto-generated — downloaded ONNX model files                   |
| `Results/`                 | Auto-generated — output `results.json` and `ranking.json`      |

---

## Running the Pipeline

From the project root, run:

```bash
python main.py
```

That's it. The pipeline will:

1. Process the reference audio and extract phrase-level pitch features
2. For each candidate track:
   - Extract vocals using the `Kim_Vocal_2.onnx` model (~24s per track on CPU)
   - Run pitch analysis via `swift-f0`
   - Slice into phrases aligned to the reference
3. Compute DTW-based similarity scores per phrase
4. Apply key-offset and SNR penalties
5. Aggregate phrase scores and rank all candidates
6. Write results to `Results/results.json` and `Results/ranking.json`

### Expected runtime

- **Per candidate track:** ~28–30 seconds (vocal separation + pitch analysis)
- **12 tracks:** ~6 minutes total
- Subsequent runs are faster if vocal stems are already cached in `assets/cleanAudioCache/`

---

## Output

| File                    | Description                                          |
|-------------------------|------------------------------------------------------|
| `Results/results.json`  | Phrase-level scores for each track (with skip labels) |
| `Results/ranking.json`  | Track-level aggregate scores and rank ordering        |

See [SOLUTION.md](SOLUTION.md) for a detailed explanation of the scoring methodology and pipeline design.
