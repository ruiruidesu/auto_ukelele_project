# ukulele-auto-tab

## Core Goal

The first core capability of this project is:

`extract a trustworthy melody note timeline from clean monophonic audio`

Only after the melody timeline looks correct should the project continue to:

- ukulele string/fret mapping
- tab rendering
- PDF export

This means the project is currently **not** optimized for:

- pretty final score output
- automatic chord-sheet generation
- broad multi-instrument transcription
- pretending that an uncertain result is a finished tab

## Current Product Scope

Current target input:

- clean monophonic `mp3` / `wav`
- ideally ukulele single-note playing, or similarly clean single-line melody audio

Current target output:

- frame-level pitch debug
- raw segmented notes
- quantized segmented notes
- segmentation quality report

Current non-goals for this stage:

- final PDF
- final ukulele mapping
- complete chord transcription

## Development Principle

The project follows this order:

1. Hear correctly
2. Segment correctly
3. Understand notation structure correctly
4. Map to ukulele correctly
5. Render correctly

If step 1 or step 2 looks wrong, do not continue to later stages.

## Inference Audit

The project now separates two different kinds of logic:

- `core inference`
  - audio preprocessing
  - pitch tracking
  - onset-aware note segmentation
  - beat and measure inference
  - rhythm-pattern decoding
- `heuristic correction`
  - repeated-note cleanup
  - anchor alignment after tuplet runs
  - closed-set pitch constraints when the user provides candidate notes

This separation exists to make overfitting visible.

If a result only becomes correct after heuristic correction, that should be treated as:

- useful progress
- but not the same thing as a fully generalized recognition ability

The pipeline now writes an `inference_audit.json` file so each run records:

- which parts are core inference
- which parts are heuristic correction
- what cross-sample validation is still missing

## Notation Understanding Layers

The project now treats tablature understanding as three distinct layers:

### 1. Page Layout Understanding

- how many tab systems exist on the page
- where each four-line ukulele system begins and ends
- where measure bars are
- where tempo and time-signature metadata belong

### 2. Symbol Recognition

- which fret number is written: `0`, `1`, `3`, etc.
- which ukulele string line the number belongs to: `A`, `E`, `C`, `G`
- which rhythmic symbol belongs to the note event:
  - stem
  - beam
  - rest
  - tie / slur / dot in future stages

### 3. Rhythm Parsing

- quarter note = `1 beat`
- eighth note = `0.5 beat`
- sixteenth note = `0.25 beat`
- each note should be assigned:
  - its measure number
  - its beat start inside the measure
  - its duration in beats

The program now exports a `notation_analysis.json` file so this layer is explicit in the pipeline.

## Ukulele Note Model

For clean single-note ukulele audio, the system treats one written note as a musical event with:

- a fresh pluck onset
- a stable pitch region after the attack
- a decay tail
- a position on the beat grid

This means:

- same pitch does not always mean the same note
- a repeated pluck of the same fret should stay as two notes if a new attack is present
- transient attack frames should not dominate final pitch choice
- beat and measure alignment must be inferred on the original time axis, not only on the trimmed waveform

Current implementation goals:

- detect note boundaries from onset evidence plus pitch continuity
- estimate pitch from the stable region, then validate with playable ukulele harmonic candidates
- keep raw note timing separate from quantized timing
- recover measure phase even when the trimmed audio starts after a rest

For short notes and dotted/mixed subdivisions, see:

- `docs/note_model.md`
  - why short notes disappear
  - how dotted-eighth / sixteenth patterns are currently modeled
  - which failure modes still belong to note extraction rather than PDF rendering
- `docs/ukulele_theory_basics.md`
  - instrument tuning and fretboard basics
  - pitch, interval, scale, chord, and rhythm foundations
  - how these theory concepts should constrain automatic ukulele transcription

## Pipeline

### 1. Audio Preprocessing

- load local audio
- convert to mono
- resample to a fixed sample rate
- trim leading/trailing silence
- normalize peak level

### 2. Melody Extraction

- frame-level pitch tracking
- onset-aware note segmentation
- fragment merging
- short-note filtering
- raw note timeline generation
- quantized note timeline generation

### 3. Quality Control

- pitch histogram
- segmentation report
- sanity warnings

### 4. Future Stages

- ukulele string/fret mapping
- text tab generation
- PDF rendering

## Current State

Implemented now:

- audio preprocessing
- frame-level pitch estimation with `librosa.pyin`
- onset-aware note segmentation
- beat-grid inference on the original time axis before trim offset is applied
- short-fragment merge/filter logic
- raw and quantized note outputs
- segmentation report generation
- sanity-check warnings

Disabled on purpose in the current stage:

- ukulele mapping as a final trusted output
- PDF generation

Still placeholder:

- `chord_sheet` mode

## Current Intermediate Files

The current melody-debug pipeline writes:

- `melody_notes_debug.txt`
- `pitch_histogram.txt`
- `segmented_notes_raw.json`
- `segmented_notes_quantized.json`
- `notation_analysis.json`
- `inference_audit.json`
- `segmentation_report.txt`
- `ukulele_mapping_debug.txt`
- `tab_preview.txt`

## What To Check First

1. `segmented_notes_raw.json`
2. `segmented_notes_quantized.json`
3. `segmentation_report.txt`

If the segmented notes do not match the melody you hear, stop there first.

## Install

```bash
.\.venv\Scripts\python -m pip install -r requirements.txt
```

## Run

```bash
.\.venv\Scripts\python main.py --mode fingerstyle_tab path\to\your\melody.mp3
```

Write outputs to a chosen folder:

```bash
.\.venv\Scripts\python main.py --mode fingerstyle_tab --output-dir output path\to\your\melody.mp3
```

## Test

```bash
.\.venv\Scripts\python -m pytest tests\test_smoke.py
```

## Next Milestones

### Milestone 1

Make `segmented_notes_raw.json` and `segmented_notes_quantized.json` reliably match the heard melody.

### Milestone 2

Re-enable ukulele mapping only after melody extraction passes manual inspection.

### Milestone 3

Generate a useful text tab preview from trusted mapping output.

### Milestone 4

Restore PDF export only after the previous three milestones are stable.
