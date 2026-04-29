# Ukulele Theory Basics

## Why this document exists

Automatic tablature generation cannot rely on pitch extraction alone.
To produce playable ukulele tabs, the system needs a shared theory baseline for:

- pitch naming
- rhythm naming
- beat grouping
- string and fret mapping
- interval and chord construction
- phrase and cadence interpretation

This document records the minimum music-theory knowledge the project should assume.

## 1. Instrument basics

### Standard tuning

The current project assumes standard re-entrant soprano/concert/tenor ukulele tuning:

- `G4`
- `C4`
- `E4`
- `A4`

In tablature, the string order is usually shown top-to-bottom as:

- `A`
- `E`
- `C`
- `G`

This means:

- top tab line = `A` string
- bottom tab line = `G` string

### Re-entrant tuning

Standard ukulele tuning uses a high `G4`, not a low `G3`.
This matters because melodic contour on the fretboard is not always the same as left-to-right pitch order on a guitar.

### Frets and semitones

Each fret raises pitch by one semitone.

Examples:

- `A string 0 = A4`
- `A string 1 = A#4 / Bb4`
- `A string 2 = B4`
- `A string 3 = C5`
- `A string 5 = D5`
- `A string 7 = E5`
- `A string 10 = G5`

## 2. Pitch theory

### Pitch names

Western pitch names cycle through:

- `C`
- `C# / Db`
- `D`
- `D# / Eb`
- `E`
- `F`
- `F# / Gb`
- `G`
- `G# / Ab`
- `A`
- `A# / Bb`
- `B`

### Octaves

The same note letter can appear in different octaves:

- `C4`
- `C5`
- `C6`

The project uses MIDI-like pitch naming because it is unambiguous for transcription.

### Intervals

The project should understand at least these interval sizes:

- minor second = 1 semitone
- major second = 2 semitones
- minor third = 3 semitones
- major third = 4 semitones
- perfect fourth = 5 semitones
- tritone = 6 semitones
- perfect fifth = 7 semitones
- minor sixth = 8 semitones
- major sixth = 9 semitones
- minor seventh = 10 semitones
- major seventh = 11 semitones
- octave = 12 semitones

This matters because melodic motion is often easier to recognize as interval steps than as isolated absolute notes.

## 3. Scale and key basics

### Major scale pattern

Whole and half steps:

- `W W H W W W H`

Example:

- `C major = C D E F G A B`

### Natural minor scale pattern

- `W H W W H W W`

Example:

- `A minor = A B C D E F G`

### Why key matters

In melodic transcription, key helps with:

- preferring diatonic notes over unstable accidentals
- choosing between close pitch candidates
- identifying likely phrase tones such as tonic and dominant

The system should not hard-force a key, but key tendency is a useful prior.

## 4. Chord basics

### Triads

Basic triad types:

- major = root + major third + perfect fifth
- minor = root + minor third + perfect fifth
- diminished = root + minor third + diminished fifth
- augmented = root + major third + augmented fifth

### Seventh chords

Useful common sevenths:

- major seventh
- dominant seventh
- minor seventh

### Why chords still matter in single-note tab

Even when output is single-note tab, chord awareness helps because:

- melody often targets chord tones on strong beats
- phrase endings often land on stable chord tones
- repeated melodic fragments may be variants over the same harmonic frame

## 5. Rhythm basics

### Meter

Meter defines:

- how many beats are in a measure
- which beats are strong or weak

Common meters:

- `4/4` = 4 quarter-note beats per measure
- `3/4` = 3 quarter-note beats per measure

### Note values

- whole note = 4 beats in `4/4`
- half note = 2 beats
- quarter note = 1 beat
- eighth note = 0.5 beat
- sixteenth note = 0.25 beat

### Dotted values

A dot adds half of the original note value.

Examples:

- dotted quarter = `1.5` beats
- dotted eighth = `0.75` beat

This is critical for transcription because many missed notes in the project come from confusing:

- a dotted value plus a short tail
- with one undivided longer note

### Tuplets

Tuplets divide a beat into equal parts that do not fit the default binary grid.

Most important here:

- eighth-note triplet = 3 equal notes inside 1 beat

Triplets should not be forced onto a plain sixteenth grid if the raw timing supports a 3-way split.

### Ties, slurs, and repeated attacks

- tie = same pitch held across note values; musically one sustained event
- slur = phrasing mark, not necessarily duration fusion
- repeated attack on the same pitch = two separate events if a fresh onset exists

This distinction is essential in ukulele transcription.
Two adjacent `7` notes are not the same as one long `7` if the player replucks.

### Pickup and silence

The first visible note in a score does not always begin at measure beat `1`.
There may be:

- leading silence
- anacrusis
- trimmed video head

So audio time `0.0` does not guarantee score beat `1.0`.

## 6. Tab reading basics

### What the number means

In tablature, the number shows fret number on the string line where it is written.

Examples:

- `A string 3 = C5`
- `E string 0 = E4`
- `C string 3 = D#4 / Eb4`
- `G string 0 = G4`

### Why the same pitch may have multiple positions

The same sounding pitch can sometimes be played in more than one location.
So automatic tablature needs two steps:

1. identify the pitch event
2. choose a likely ukulele position

### Playability preferences

For beginner-friendly single-note tabs, prefer:

- low fret positions when possible
- short hand shifts
- consistent local position for repeated phrases
- string choices that preserve melodic contour cleanly

## 7. What this means for automatic transcription

The project should reason in this order:

1. Did a new note event start
2. What is the note pitch
3. How long does it last
4. Where does it sit in the beat and measure
5. Is it part of a dotted pattern, triplet, or binary subdivision
6. Which ukulele string and fret best realize it

## 8. Specific failure modes to watch for

### Short-note loss

Common cause:

- short note has weak sustain
- later longer note dominates frame voting

### Same-pitch reattack collapse

Common cause:

- two adjacent same-pitch notes are merged because the system misses the second onset

### Wrong beat layer

Common cause:

- the system chooses double-time or half-time BPM
- all later note grouping becomes wrong

### Repeated phrase inconsistency

Common cause:

- repeated musical phrases are decoded locally instead of using phrase-level consensus

## 9. Practical theory checks for this repo

Before accepting a generated tab, check:

- are repeated same-pitch plucks still separate where they should be
- are dotted-eighth/sixteenth pairs preserved
- are triplets kept as triplets instead of flattened into binary divisions
- does measure count match the musical phrase structure
- does string/fret choice stay playable
- do repeated phrases stay musically consistent

## 10. Current project assumptions

For now, the project is still optimized for:

- clean monophonic audio
- mostly single-note ukulele melody
- standard `gCEA` tuning

It is not yet a full general music-transcription engine.
