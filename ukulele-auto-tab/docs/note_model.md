# Ukulele Single-Note Model

## Why this exists

The project should not "match a benchmark tab by luck".
It should explain *why* a few notes align to the score:

1. a new pluck happened
2. that pluck settled on a stable pitch
3. the pitch belongs to a written note event
4. the note event belongs to a beat location in the meter

## Three notation layers

To make tablature render faithfully, the program needs to model three separate layers.

### Layer 1: page layout understanding

The system must know:

- how many four-line tab systems are on the page
- which measures belong to each system
- where measure lines, tempo, and time signature belong

This is the layout layer. It answers "where on the page should this information live?"

### Layer 2: symbol recognition

The system must know, for each note event:

- what fret number is written
- which ukulele string line it belongs to
- which rhythmic mark belongs to it:
  - stem
  - beam
  - rest
  - tie/slur/dot in future stages

This is the symbol layer. It answers "what musical symbol is this?"

### Layer 3: rhythm parsing

The system must then translate visual rhythm into musical duration:

- quarter note = 1 beat
- eighth note = 0.5 beat
- sixteenth note = 0.25 beat

Given a meter such as `4/4`, the system must derive:

- measure number
- beat start inside the measure
- duration in beats

This is the rhythm layer. It answers "when does this note happen and how long does it last?"

## Event definition

For clean monophonic ukulele audio, one note is modeled as:

- `attack`: a new pluck onset
- `stable region`: the part used to decide the note pitch
- `decay`: the tail that should usually remain inside the same note
- `grid placement`: the note's rhythmic position after note timing is considered trustworthy

## Practical rules

### 1. Same pitch can still be two notes

If two adjacent segments share the same MIDI pitch but the waveform shows a fresh attack,
they should remain as two plucked notes.

This is why same-pitch merging must be conservative.

### 2. Attack should not dominate pitch identity

The attack of a plucked string can briefly over-emphasize harmonics.
Pitch selection should prefer the stable region after the first transient frames.

### 3. Pitch continuity can repair missed onsets

If onset detection misses a boundary but the pitch changes and stays changed,
the segment should still split into two note events.

### 4. Measure alignment belongs to the original timeline

If trim removes leading silence, the trimmed waveform no longer starts at the musical barline.
Beat phase and measure boundaries should be inferred on the original timeline, then shifted into the trimmed view.

## Short-note and dotted-subdivision model

This project now treats very short notes as a separate recognition problem instead of assuming
that every note has a long stable sustain.

### Why very short notes are hard

Short notes are easy to lose because all of these can happen at once:

- the new pluck has a weak attack
- the note only produces one or two clearly voiced frames
- the following note is louder and longer
- frame-level pitch tracking prefers the later stable note
- post-processing merges the short event into the following anchor note

This is why a short tail note in patterns such as `front_eighth_then_sixteenth` or
`dotted_eighth_then_sixteenth` often disappears if the system only trusts stable sustain.

### What the system currently uses to keep short notes

The pipeline now tries to preserve short notes with these rules:

- `onset evidence first`
  - a short note is allowed to exist if the attack looks like a real new pluck
- `attack-local pitch for short notes`
  - short-note pitch is estimated from the first local attack region, not only from the later sustain
- `conservative merge`
  - short notes with onset support are less likely to be merged into the following longer note
- `pattern-aware decoding`
  - after note events exist, beat-local decoding compares templates such as:
    - `front_sixteenth_then_eighth`
    - `front_eighth_then_sixteenth`
    - `dotted_eighth_then_sixteenth`
    - `sixteenth_then_dotted_eighth`

### How dotted patterns are interpreted

The symbolic layer treats these patterns as beat-local structures:

- `dotted_eighth_then_sixteenth`
  - one longer note lasting `0.75` beat, followed by a short note lasting `0.25` beat
- `sixteenth_then_dotted_eighth`
  - one short note lasting `0.25` beat, followed by a longer note lasting `0.75` beat
- `front_sixteenth_then_eighth`
  - three-note pattern with relative beat offsets like `0.0, 0.25, 0.5`
- `front_eighth_then_sixteenth`
  - three-note pattern with relative beat offsets like `0.0, 0.5, 0.75`

These are not recognized from notation graphics alone. The program first tries to recover
the audio note events, then checks whether their raw timing and quantized timing better match
one of these rhythmic templates.

### Why a missing short note usually means an earlier failure

If the final PDF is missing the last short note in a dotted or mixed subdivision pattern,
the failure usually happened before rendering:

1. the short note never became a separate note event
2. the short note became a note event, but its pitch was pulled toward the neighboring long note
3. the short note survived note extraction, but beat-level decoding collapsed the pattern into a simpler shape

So the right debugging order is:

1. inspect raw segmented notes
2. inspect quantized note events
3. inspect beat pattern decoding
4. inspect PDF rendering only after the first three look correct

## Known failure modes

The system is still weak in these situations:

- repeated short-note phrases where the same melody appears multiple times but one repetition loses a short note
- long passages where beat-level decoding is correct locally but inconsistent across repeated measures
- very high-register short notes with weak attack and little stable sustain
- cases where the global BPM/meter hypothesis is slightly wrong, because beat misalignment can erase short-note slots

These failures should be treated as note-extraction or rhythm-decoding problems, not as final-PDF problems.

## Current implementation mapping

- onset detection: `librosa.onset`
- frame pitch: `librosa.pyin`
- in-segment stable pitch choice: weighted frame vote + playable ukulele harmonic candidate check
- note segmentation: onset windows + pitch continuity split
- note merge/filter: conservative same-pitch merge, short-fragment filter
- rhythm grid: beat tracking on the original audio time axis, then offset into the trimmed note timeline
- notation layer: symbolic note events now include string line, fret number, beat start, duration, rhythm value, and beam grouping

## What success looks like

Before ukulele mapping resumes, a benchmark passage should satisfy:

- repeated plucks stay repeated
- long notes do not fragment into many tiny notes
- short ornamental tails do not become fake notes
- measure phase is plausible for the reference score
- the extracted note contour matches the heard melody without manual score-specific hacks
