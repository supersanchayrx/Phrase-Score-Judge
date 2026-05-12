# Solution — Vocal Pitch Scoring Pipeline

> **Setup & Run Instructions:** See [INSTRUCTIONS.md](INSTRUCTIONS.md) for dependencies, directory layout, and how to run the pipeline.

## Overview

This pipeline scores candidate vocal tracks against a reference performance at the phrase level. It extracts isolated vocals from noisy candidate audio, computes pitch features using `swift-f0`, aligns them to the reference via Dynamic Time Warping, normalizes for key differences, penalizes noise, and produces a ranked list of candidates.

---

## Pipeline Architecture

The system is organized into five stages, each handled by a dedicated module:

```
Reference Audio ──► Phrase Slicing ──► Feature Extraction ──────────────────────────────┐
                    (struct.json)       (swift-f0)                                       │
                                                                                         ▼
Candidate Audio ──► Vocal Separation ──► Feature Extraction ──► Alignment & Offset ──► DTW Scoring ──► Ranking
                    (Kim_Vocal_2)        (swift-f0)              Correction               (librosa)
```

| Module                  | Responsibility                                                                 |
|-------------------------|--------------------------------------------------------------------------------|
| `PreProcessRefAudio.py` | Slices reference audio into 4 phrase windows using `struct.json`, extracts pitch features |
| `VocalsExtractor.py`    | Isolates vocals from candidate tracks using the `Kim_Vocal_2.onnx` model; computes SNR from vocal/instrumental stems |
| `ProcessAudioArray.py`  | Runs `swift-f0` pitch detection on any audio array; detects voice onset time (VAD) |
| `PreProcessCandidates.py` | Orchestrates vocal extraction → feature extraction → phrase slicing for every candidate |
| `dtw.py`                | Computes phrase-level melodic similarity via Dynamic Time Warping              |
| `main.py`               | Runs the full pipeline, applies penalties, aggregates scores, and writes results |

---

## Step-by-Step Methodology

### 1. Reference Audio Segmentation

The reference vocal track (`assets/reference/vocals.mp3`) is segmented into **4 phrases** using the timing metadata provided in `struct.json`. Each phrase's `start_time_ms` and `end_time_ms` defines a window. The entire 4-phrase region is extracted as a single contiguous block and passed through `swift-f0` once, then the resulting frame-level features are sliced into individual phrase windows by matching frame timestamps to the phrase boundaries.

> **Note:** This approach could also be done without `struct.json` by detecting voice activity boundaries (VAD-based segmentation), but the provided metadata made it straightforward.

### 2. Feature Extraction (`swift-f0`)

For both reference and candidate audio, the [`swift-f0`](https://github.com/lars76/swift-f0) library handles pitch analysis. From each audio buffer it extracts:

- **Fundamental frequency (F0):** The detected pitch in Hz for each analysis frame.
- **Confidence:** `swift-f0`'s detection confidence level for each frame — how certain the model is that the detected pitch is correct.
- **Voicing (VAD):** A boolean per frame indicating whether the frame contains voiced speech, also provided by `swift-f0`'s analysis results.
- **Frame timestamps:** The time position of each analysis frame, determined by the internal buffer size that constitutes one frame in `swift-f0`'s analysis.

Pitch values are converted from Hz to **cents** (relative to A440) using the standard formula:

```
cents = 1200 × log₂(f0 / 440)
```

Only frames where `f0 > 0` and `voicing = true` receive a valid cent value; all others are set to `NaN`.

### 3. Vocal Isolation (Candidate Pre-Processing)

The candidate tracks contain significant noise, background music, and instrumentation — unlike the clean reference vocal. Running pitch analysis directly on raw candidates produced results that were far outside any reasonable bounds.

#### Initial attempt — STFT-based spectral gating

An initial denoising pipeline was built using Short-Time Fourier Transform (STFT) on audio chunks. A noise profile was estimated from a short leading segment, and a spectral mask was applied. This was fast and produced noticeably cleaner audio, but the noise mask was inconsistent across different tracks — the method reduced noise but could not reliably isolate the vocal from background instrumentation. What was needed was not just denoised audio, but **isolated vocals** — essentially an acapella extraction.

#### Final approach — Neural source separation

After evaluating several libraries (including Facebook's Demucs, which was either insufficiently accurate or too computationally expensive for this use case), the [`audio-separator`](https://github.com/nomadkaraoke/python-audio-separator) library was selected with the **`Kim_Vocal_2.onnx`** model. This model performs vocal/instrumental source separation and runs in near real-time: a ~24-second candidate track is processed in approximately 24 seconds on a mid-range laptop (Intel i5-1200UH + RTX 3050, CPU-only inference — no CUDA acceleration). The STFT-based denoising pipeline was scrapped entirely in favor of this approach.

The separator outputs two stems per track:
- **Vocals stem** — used for pitch analysis
- **Instrumental stem** — retained for SNR calculation (see §6)

Extracted stems are cached to `assets/cleanAudioCache/` so subsequent runs skip the separation step.

### 4. Candidate Alignment & Phrase Slicing

The reference audio and candidate recordings are **not temporally synchronized** — candidates may begin singing at different offsets (some start at ~0.5s, others at ~0.7s or ~0.8s into the file). Simple index-based comparison is therefore not possible.

To handle this, a **voice onset time** is computed for each candidate during feature extraction (`ProcessAudioArray.py`). The algorithm scans the VAD and confidence arrays for the first window of `N` consecutive voiced frames (default `N = 5`) where:
1. All frames are marked as voiced
2. Mean confidence exceeds a threshold (0.6)
3. RMS energy of the corresponding audio buffer exceeds a minimum (0.001) to filter out false positives from silence

This voice onset time is then combined with the reference phrase boundaries from `struct.json` to compute approximate phrase windows for the candidate:

```
candidate_phrase_start = voice_onset + (ref_phrase_start − ref_first_phrase_start) / 1000
candidate_phrase_end   = voice_onset + (ref_phrase_end   − ref_first_phrase_start) / 1000
```

The full candidate audio is analyzed by `swift-f0` once (for the entire file), and the frame-level feature arrays are then sliced into per-phrase segments using the computed window boundaries.

### 5. Melodic Similarity — Dynamic Time Warping (DTW)

Even after alignment, the reference and candidate phrase contours are not frame-for-frame synchronized — candidates may sing slightly faster, slower, or with different rhythmic phrasing. A straightforward frame-by-frame correlation would fail here.

**Dynamic Time Warping (DTW)** solves this by finding an optimal alignment between two sequences that may vary in speed. DTW exploits small time-temporal distortions and correlates data values accordingly, effectively "warping" the time axis of one sequence to best match the other. The underlying math is straightforward — it finds the minimum-cost path through a cost matrix using dynamic programming.

#### Cost matrix construction

A pairwise cost matrix is constructed between the voiced cent values of the reference and candidate phrases. The cost between any two frames is defined as the **absolute difference** in their cent values — identical pitches yield a cost of 0, and differing pitches yield a cost proportional to how far apart they are:

```
cost[i][j] = |ref_cents[i] − cand_cents[j]|
```

Only voiced frames (non-`NaN` cent values) are included; unvoiced frames are excluded from the comparison.

#### DTW computation

The cost matrix is passed to **`librosa.sequence.dtw`**, which returns the accumulated cost matrix and the optimal warp path. The normalized DTW distance is computed as:

```
normalized_distance = accumulated_cost[-1, -1] / len(warp_path)
```

This is converted to a 0–100 score:

```
score = max(0, 100 × (1 − normalized_distance / 300))
```

where 300 cents is treated as the maximum reasonable deviation (approximately a minor third). A perfect melodic match scores 100; deviations beyond 300 cents per frame on average score 0.

### 6. Melody Normalization & Key-Offset Penalty

#### The problem

Initial DTW scoring produced reasonable results for candidates that sang in the same key as the reference. However, when tested against recordings from YouTube covers and personal recordings that were melodically accurate but sung in a different key, the scores were disproportionately low — clean, coherent performances were ranked below noisy tracks simply because they were transposed.

#### The fix — Melody normalization

Before DTW comparison, both reference and candidate pitch contours are **normalized to a zero-median baseline**:

```
normalized_cents = cents − median(cents)    [per phrase]
```

This removes the absolute pitch level and compares only the **melodic shape** — the pattern of intervals and movement. A candidate singing the correct melody in a different key will now align well with the reference.

#### Key-offset penalty

Melody normalization enables fair comparison regardless of key, but singing in the correct key *should* still be rewarded. To balance this, a **key-offset penalty** captures how far the candidate's median pitch is from the reference's median pitch:

```
key_offset = |ref_median_cents − cand_median_cents|
key_penalty = max(0, 100 × (1 − key_offset / 400))
```

A candidate singing in the same key gets a full 100 for this component; one that is 400+ cents away (roughly a major third) gets 0.

---

## Scoring Definition

The final phrase score is a weighted blend of melodic shape accuracy (DTW) and key fidelity, further modulated by an audio quality (SNR) penalty:

```
phrase_score = (0.75 × dtw_score + 0.25 × key_penalty) × snr_penalty
```

| Component        | Weight | What it measures                                    | Range  |
|------------------|--------|-----------------------------------------------------|--------|
| `dtw_score`      | 75%    | Melodic contour similarity (shape, intervals)       | 0–100  |
| `key_penalty`    | 25%    | How close the candidate's key is to the reference   | 0–100  |
| `snr_penalty`    | ×1.0   | Audio quality multiplier (caps noisy tracks)        | 0–1    |

### SNR Penalty

The vocal extraction model does a decent job of isolating vocals, but residual artifacts of background noise remain in the vocal stem. These artifacts gave noisy tracks a misleading advantage — the residual noise in the extracted vocal occasionally created pitch contours that happened to correlate with the reference by coincidence.

To counteract this, a **Signal-to-Noise Ratio (SNR) penalty** is applied at the track level. The vocal separator already outputs both a vocal stem and an instrumental stem, so the SNR is computed directly from these:

```
SNR = RMS(vocals) / RMS(instrumental)
```

If the SNR falls below a threshold (1.75), the phrase score is scaled down proportionally:

```
snr_penalty = min(1.0, SNR / 1.75)
```

Clean tracks with a dominant vocal stem are unaffected; tracks where the instrumental energy rivals or exceeds the vocal energy are penalized.

### Edge Cases & Skip Conditions

Phrases or entire tracks are marked as `skipped` (with a label and reason) when:

| Label                     | Condition                                                    |
|---------------------------|--------------------------------------------------------------|
| `silent_track`            | No voice activity detected anywhere in the track             |
| `missing_phrase`          | Candidate audio is shorter than expected / phrase not found   |
| `insufficient_voiced`     | Fewer than 10 voiced frames in the phrase                    |
| `pitch_extraction_failed` | DTW could not be computed (empty voiced contour after filtering) |

---

## Ranking

1. **Per-phrase scores** are computed as described above for each of the 4 phrases.
2. The **aggregate track score** is the arithmetic mean of all scored (non-skipped) phrases.
3. Tracks are sorted by aggregate score in descending order. The highest score receives rank 1.
4. Tracks with no scorable phrases receive `aggregate_score: null` and `rank: null`.

Results are written to:
- `Results/results.json` — phrase-level detail (per-track with nested phrase records)
- `Results/ranking.json` — track-level ranking summary

---

## Tradeoffs & Limitations

### Processing Time

Each candidate track requires ~24 seconds for vocal separation (real-time factor on CPU) plus a few seconds for `swift-f0` analysis, totaling **~28–30 seconds per candidate**. For the 12-track dataset this is manageable (~6 minutes total), but scaling to hundreds of tracks without GPU acceleration would be prohibitive.

### Vocal Separation Quality

The `Kim_Vocal_2.onnx` model is effective but imperfect. Residual instrumental artifacts in the vocal stem occasionally influence pitch detection. The SNR penalty mitigates this but is a coarse correction — it operates at the track level rather than the frame level.

### Alignment Assumptions

The phrase slicing for candidates assumes that the singer follows roughly the same temporal structure as the reference (same phrase durations, same song section). Candidates who deviate significantly in timing — e.g., adding ad-libs, pausing mid-phrase, or singing at a very different tempo — may have their phrases misaligned, leading to inaccurate scores.

### Fixed Hyperparameters

Several thresholds are hand-tuned for this specific dataset:
- `maxKeyOffset = 400 cents` (key penalty cap)
- `snrThreshold = 1.75` (noise penalty threshold)
- `maxReasonableDistance = 300 cents` (DTW score ceiling)
- `phraseVsKeyRatio = 0.75` (DTW vs. key weight)

These may not generalize well to different songs, vocal ranges, or recording conditions without re-tuning.

### Single-Song Scope

The pipeline is built around a single reference track and its `struct.json`. Extending to multiple songs would require either a generalized VAD-based phrase detector or per-song metadata files.

---

## Potential Improvements (AI-Analyzed Suggestions)

> The following suggestions were generated after an AI analyzed the codebase and are not part of the original implementation.

### Pitch Analysis & Feature Engineering
- **Chroma features instead of raw cents:** Using chroma (pitch-class) representations would make the comparison inherently octave-invariant, eliminating cases where a candidate sings the correct note in a different octave and gets penalized.
- **Vibrato and timing metrics:** Incorporating additional vocal quality features — such as vibrato rate/extent, rhythmic precision, or onset sharpness — would produce a more holistic score beyond pure pitch accuracy.
- **Frame-level confidence weighting:** Currently all voiced frames are weighted equally in DTW. Weighting frames by `swift-f0`'s confidence score would reduce the influence of uncertain pitch estimates on the final score.

### Alignment & Segmentation
- **Adaptive phrase boundary detection:** Rather than projecting reference phrase boundaries onto candidates using a single voice-onset offset, a more robust approach would use audio fingerprinting or onset detection to independently locate phrase boundaries in each candidate.
- **VAD-based automatic segmentation:** For tracks without `struct.json`, implementing Voice Activity Detection (e.g., via `silero-vad` or energy-based methods) to automatically segment phrases would make the pipeline fully metadata-independent.

### Noise Handling
- **Frame-level SNR penalty:** Instead of applying a single track-level SNR multiplier, computing SNR on a per-phrase or per-frame basis would more precisely penalize sections that are noisy while preserving scores for clean sections within the same track.
- **Spectral noise fingerprinting:** Identifying the specific frequency bands contaminated by residual artifacts and down-weighting those frames in the DTW comparison would be more surgical than a blanket energy ratio.

### Scoring Methodology
- **Learned scoring weights:** The current weights (0.75/0.25 for DTW/key) are hand-picked. Training a small regression model on human-annotated quality ratings would let the system learn optimal weights for each feature.
- **Percentile-based normalization:** Converting raw scores to percentile ranks within a dataset would make scores more interpretable and comparable across different songs or recording conditions.
- **Confidence interval reporting:** Reporting a confidence range alongside each score (derived from per-frame variance or cross-phrase consistency) would better communicate score reliability.

### Performance & Scalability
- **GPU-accelerated vocal separation:** Enabling CUDA support for the ONNX model (or switching to a TensorRT-optimized variant) would reduce the ~24s per-track separation time to a few seconds.
- **Parallel processing:** The current pipeline processes candidates sequentially. Since each candidate is independent, parallelizing across tracks (e.g., via `multiprocessing` or `concurrent.futures`) would yield near-linear speedup on multi-core machines.
- **Incremental caching:** The vocal separation cache exists, but feature extraction results (cents, VAD, confidence arrays) could also be cached to avoid re-running `swift-f0` on unchanged inputs.
