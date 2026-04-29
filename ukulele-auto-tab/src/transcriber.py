from __future__ import annotations

import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
import librosa
import numpy as np
try:
    import aubio
except ImportError:  # pragma: no cover - optional tempo backend
    aubio = None

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
except ImportError:  # pragma: no cover - optional dependency at runtime
    A4 = None
    canvas = None

from src.schemas import (
    AudioPreprocessSummary,
    ChordAnchor,
    ChordSection,
    ChordSheetSchema,
    InferenceComponent,
    LyricLine,
    MelodyNote,
    MelodyPipelineResult,
    RhythmAnalysis,
    SourceClassification,
    TabNotationAnalysis,
    TabPageLayout,
    TabSymbolEvent,
    TabSystemLayout,
    UkuleleMappedNote,
    ValidationObservation,
)

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
UKULELE_STRINGS: list[tuple[str, int, int]] = [
    ("A", 1, 69),
    ("E", 2, 64),
    ("C", 3, 60),
    ("G", 4, 67),
]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"}
UKULELE_NOTE_MIN = "C4"
UKULELE_NOTE_MAX = "A5"
HOP_LENGTH = 128
UKULELE_PLAYABLE_MIDI = sorted(
    {
        open_midi + fret
        for _, _, open_midi in UKULELE_STRINGS
        for fret in range(0, 13)
    }
)
PDF_STRING_ORDER = ["A", "E", "C", "G"]
BEAT_HOP_LENGTH = 256


def _core_inference_components() -> list[InferenceComponent]:
    return [
        InferenceComponent(
            name="audio_preprocessing",
            category="core_inference",
            description="Load audio/video, extract mono audio, resample, trim silence, and normalize level.",
            generalization_note="General-purpose preprocessing that does not depend on a specific song pattern.",
        ),
        InferenceComponent(
            name="frame_level_pitch_tracking",
            category="core_inference",
            description="Estimate frame-level F0 candidates and voiced/unvoiced states from the waveform.",
            generalization_note="Core monophonic pitch extraction; intended to generalize across clean single-note inputs.",
        ),
        InferenceComponent(
            name="onset_aware_note_segmentation",
            category="core_inference",
            description="Convert frame-level pitch into raw note events using onset evidence, pitch continuity, and minimum duration logic.",
            generalization_note="General note-boundary inference for plucked single-note audio.",
        ),
        InferenceComponent(
            name="beat_and_measure_inference",
            category="core_inference",
            description="Estimate BPM, beat grid, measure boundaries, and base quantization phase from audio timing evidence.",
            generalization_note="General rhythmic inference layer, but still imperfect for complex subdivisions.",
        ),
        InferenceComponent(
            name="pattern_level_rhythm_decoding",
            category="core_inference",
            description="Decode beat-local rhythm patterns such as quarter, eighth-pair, sixteenth groups, and triplets from quantized note timing.",
            generalization_note="General symbolic rhythm interpretation, but sensitive to upstream segmentation quality.",
        ),
    ]


def _heuristic_correction_components() -> list[InferenceComponent]:
    return [
        InferenceComponent(
            name="closed_set_pitch_constraint",
            category="heuristic_correction",
            description="Restrict candidate pitches to a known note set when the caller provides candidate notes.",
            generalization_note="Useful for benchmarked exercises, but not a general open-set recognition ability.",
        ),
        InferenceComponent(
            name="same_pitch_fragment_consolidation",
            category="heuristic_correction",
            description="Merge or preserve same-pitch neighboring fragments based on confidence, gap size, and repeated-run context.",
            generalization_note="Generalizable heuristic for plucked repeated notes, but still threshold-driven.",
        ),
        InferenceComponent(
            name="stable_anchor_alignment_after_tuplet_run",
            category="heuristic_correction",
            description="After a same-pitch tuplet run, prefer aligning a following different-pitch stable anchor to the next strong beat when the raw timing supports that interpretation.",
            generalization_note="Abstracted from case-specific failures, but intended as a general rule for repeated-run to anchor transitions.",
        ),
        InferenceComponent(
            name="anchor_sequence_alignment_after_repeated_tuplets",
            category="heuristic_correction",
            description="When repeated same-pitch tuplets are followed by a sequence of long anchors with consistent raw spacing, promote those anchors to consecutive integer beats.",
            generalization_note="Generalized from repeated-run benchmark failures; still a heuristic correction rather than pure model inference.",
        ),
        InferenceComponent(
            name="repeated_note_rhythm_regularization",
            category="heuristic_correction",
            description="Regularize dense same-pitch note runs toward a more consistent symbolic rhythm when raw timing strongly suggests a repeated pattern.",
            generalization_note="General post-processing heuristic that can help or overfit depending on the sample.",
        ),
        InferenceComponent(
            name="pitch_path_optimization",
            category="heuristic_correction",
            description="Use a note-sequence path search to prefer pitch candidates that fit the observed frequency while keeping neighboring transitions musically plausible.",
            generalization_note="General smoothing heuristic for open-set single-note material; more robust than per-note greedy assignment, but still not a learned model.",
        ),
        InferenceComponent(
            name="repeated_phrase_consensus_regularization",
            category="heuristic_correction",
            description="Detect strongly similar one- or two-measure phrases and nudge their symbolic beat positions and durations toward a shared consensus template.",
            generalization_note="General phrase-consistency heuristic intended to stabilize repeated melodies, though still threshold-driven and conservative.",
        ),
    ]


def _build_validation_observations(
    candidate_note_names: list[str] | None,
) -> list[ValidationObservation]:
    observations = [
        ValidationObservation(
            sample_name="case3",
            validation_type="benchmark_fit",
            status="partial",
            notes="Current rules were actively tuned against case3 to fix triplet-to-anchor transitions. This improves benchmark fit but does not by itself prove generalization.",
        ),
        ValidationObservation(
            sample_name="case1",
            validation_type="cross_sample_triplet_check",
            status="not_available",
            notes="No matching unseen local sample with a similar triplet-plus-anchor structure has been validated yet.",
        ),
        ValidationObservation(
            sample_name="case2",
            validation_type="cross_sample_regression_check",
            status="pending",
            notes="Case2 is rhythmically different from case3. It is still useful as a regression check to ensure new heuristics do not break non-triplet repeated-note material.",
        ),
    ]
    if candidate_note_names:
        observations.append(
            ValidationObservation(
                sample_name="closed_set_mode",
                validation_type="constraint_notice",
                status="active",
                notes=f"Closed-set candidate notes are active: {', '.join(candidate_note_names)}. Pitch recognition results are not open-set in this run.",
            )
        )
    return observations


def read_audio_file(file_path: str | Path) -> Path:
    audio_path = Path(file_path)
    print(f"reading audio... {audio_path}")
    return audio_path


def build_chord_sheet_schema(audio_path: str | Path) -> dict[str, object]:
    path = Path(audio_path)
    print(f"building chord sheet schema... {path}")
    schema = ChordSheetSchema(
        output_type="placeholder_sheet_structure",
        schema_name="chord_sheet_schema",
        source_path=str(path),
        notes="This remains a placeholder chord-sheet structure. Real chord transcription is not implemented in this stage.",
        sections=[
            ChordSection(
                name="Verse 1",
                lines=[
                    LyricLine(
                        text="[placeholder lyric line 1]",
                        chord_anchors=[
                            ChordAnchor(beat_offset=0.0, lyric_char_index=0, chord_name="C"),
                            ChordAnchor(beat_offset=2.0, lyric_char_index=12, chord_name="G"),
                        ],
                    )
                ],
            )
        ],
    )
    return schema.to_dict()


def _format_note_name(midi_note: int) -> str:
    octave = midi_note // 12 - 1
    return f"{PITCH_CLASSES[midi_note % 12]}{octave}"


def _parse_note_name(note_name: str) -> int:
    return int(round(librosa.note_to_midi(note_name)))


def _safe_mean(values: np.ndarray | list[float], default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    return float(np.mean(np.asarray(values, dtype=float)))


def _safe_median(values: np.ndarray | list[float], default: float = 0.0) -> float:
    if len(values) == 0:
        return default
    return float(np.median(np.asarray(values, dtype=float)))


def _frame_window(
    frame_pitch_debug: list[dict[str, object]],
    start_time: float,
    end_time: float,
) -> list[dict[str, object]]:
    return [
        frame
        for frame in frame_pitch_debug
        if start_time <= float(frame["time"]) <= end_time
    ]


def _guess_source_classification(
    audio: np.ndarray,
    sample_rate: int,
    frame_pitch_debug: list[dict[str, object]],
    notes: list[MelodyNote],
) -> SourceClassification:
    duration_seconds = max(float(librosa.get_duration(y=audio, sr=sample_rate)), 1e-6)
    harmonic_audio, percussive_audio = librosa.effects.hpss(audio)
    harmonic_energy = float(np.sum(np.square(harmonic_audio))) + 1e-6
    percussive_energy = float(np.sum(np.square(percussive_audio))) + 1e-6
    harmonic_ratio = harmonic_energy / percussive_energy

    centroid = _safe_mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate)[0])
    rolloff = _safe_mean(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate)[0])
    flatness = _safe_mean(librosa.feature.spectral_flatness(y=audio)[0])
    zcr = _safe_mean(librosa.feature.zero_crossing_rate(y=audio)[0])
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sample_rate)
    active_pitch_classes = float(
        np.mean(np.sum(chroma >= 0.45, axis=0))
    ) if chroma.size else 0.0

    voiced_frames = [frame for frame in frame_pitch_debug if frame.get("voiced")]
    voiced_ratio = len(voiced_frames) / max(len(frame_pitch_debug), 1)
    onset_rate = sum(1 for frame in frame_pitch_debug if frame.get("onset_peak")) / duration_seconds
    midi_values = [note.midi for note in notes]
    median_midi = _safe_median(midi_values, default=64.0)
    pitch_span = (max(midi_values) - min(midi_values)) if midi_values else 0
    note_density = len(notes) / duration_seconds
    median_duration = _safe_median([note.duration for note in notes], default=0.24)
    ukulele_playable_ratio = (
        float(np.mean([note.midi in UKULELE_PLAYABLE_MIDI for note in notes]))
        if notes
        else 0.0
    )

    family_scores: dict[str, float] = {
        "ukulele": 0.0,
        "guitar": 0.0,
        "piano": 0.0,
        "vocal": 0.0,
        "mixed": 0.0,
    }
    family_scores["ukulele"] += 1.1 if harmonic_ratio >= 1.3 else -0.2
    family_scores["ukulele"] += 1.0 if 60 <= median_midi <= 79 else -0.4
    family_scores["ukulele"] += 0.7 if 650 <= centroid <= 3200 else -0.2
    family_scores["ukulele"] += 0.6 if 0.55 <= voiced_ratio <= 0.98 else -0.2
    family_scores["ukulele"] += 0.4 if active_pitch_classes <= 2.2 else -0.3
    family_scores["ukulele"] += 0.5 if 1.2 <= note_density <= 9.0 else 0.0
    family_scores["ukulele"] += 0.7 if median_duration <= 0.32 else 0.0
    family_scores["ukulele"] += 0.6 if ukulele_playable_ratio >= 0.92 else 0.0

    family_scores["guitar"] += 1.0 if harmonic_ratio >= 1.15 else 0.0
    family_scores["guitar"] += 1.0 if 48 <= median_midi <= 76 else -0.2
    family_scores["guitar"] += 0.7 if centroid <= 2600 else -0.3
    family_scores["guitar"] += 0.4 if active_pitch_classes >= 1.6 else 0.0
    family_scores["guitar"] += 0.3 if pitch_span >= 7 else 0.0
    family_scores["guitar"] -= 0.45 if median_duration <= 0.28 and median_midi >= 60 else 0.0
    family_scores["guitar"] -= 0.35 if ukulele_playable_ratio >= 0.96 else 0.0

    family_scores["piano"] += 1.1 if percussive_energy > harmonic_energy * 0.7 else 0.0
    family_scores["piano"] += 0.8 if rolloff >= 2600 else 0.0
    family_scores["piano"] += 0.4 if onset_rate >= 2.5 else 0.0
    family_scores["piano"] += 0.4 if active_pitch_classes >= 1.5 else 0.0

    family_scores["vocal"] += 1.1 if zcr >= 0.085 else 0.0
    family_scores["vocal"] += 0.8 if flatness >= 0.035 else 0.0
    family_scores["vocal"] += 0.6 if onset_rate <= 3.0 else 0.0
    family_scores["vocal"] += 0.4 if 55 <= median_midi <= 84 else 0.0

    family_scores["mixed"] += 1.3 if active_pitch_classes >= 2.2 else 0.0
    family_scores["mixed"] += 0.8 if voiced_ratio <= 0.55 else 0.0
    family_scores["mixed"] += 0.5 if pitch_span >= 12 else 0.0

    ordered_families = sorted(family_scores.items(), key=lambda item: item[1], reverse=True)
    source_family, top_score = ordered_families[0]
    second_score = ordered_families[1][1] if len(ordered_families) > 1 else top_score
    source_confidence = round(max(0.35, min(0.98, 0.55 + (top_score - second_score) / 3.5)), 4)

    if active_pitch_classes <= 1.35 and voiced_ratio >= 0.55 and harmonic_ratio >= 1.0:
        texture_type = "monophonic_clean"
        texture_confidence = 0.82
    elif active_pitch_classes <= 2.25:
        texture_type = "polyphonic_with_main_melody"
        texture_confidence = 0.68
    else:
        texture_type = "polyphonic_mixed_uncertain"
        texture_confidence = 0.56

    vocal_presence = source_family == "vocal" or (zcr >= 0.1 and flatness >= 0.04)
    if source_family == "ukulele" and texture_type == "monophonic_clean":
        recommended_pipeline = "ukulele_monophonic_pipeline"
    elif source_family == "ukulele":
        recommended_pipeline = "ukulele_real_performance_pipeline"
    elif texture_type == "monophonic_clean":
        recommended_pipeline = "external_monophonic_to_ukulele_pipeline"
    else:
        recommended_pipeline = "polyphonic_melody_extraction_pipeline"

    notes_text = (
        f"family={source_family}, texture={texture_type}, harmonic_ratio={harmonic_ratio:.2f}, "
        f"centroid={centroid:.1f}Hz, active_pitch_classes={active_pitch_classes:.2f}, "
        f"voiced_ratio={voiced_ratio:.2f}, median_duration={median_duration:.3f}s, "
        f"ukulele_playable_ratio={ukulele_playable_ratio:.2f}. This is a heuristic source router, not a trained classifier."
    )
    return SourceClassification(
        source_family=source_family,
        source_confidence=source_confidence,
        texture_type=texture_type,
        texture_confidence=round(texture_confidence, 4),
        vocal_presence=bool(vocal_presence),
        recommended_pipeline=recommended_pipeline,
        notes=notes_text,
    )


def _best_same_string_pair(first_midi: int, second_midi: int) -> tuple[str, int, int] | None:
    first_candidates = _candidate_positions(first_midi)
    second_candidates = _candidate_positions(second_midi)
    shared_pairs: list[tuple[str, int, int]] = []
    for first_string, _, first_fret in first_candidates:
        for second_string, _, second_fret in second_candidates:
            if first_string != second_string:
                continue
            shared_pairs.append((first_string, first_fret, second_fret))
    if not shared_pairs:
        return None
    return min(
        shared_pairs,
        key=lambda item: (abs(item[2] - item[1]), max(item[1], item[2]), item[1]),
    )


def _monotonic_glide_score(
    frame_pitch_debug: list[dict[str, object]],
    start_time: float,
    end_time: float,
    start_midi: int,
    end_midi: int,
) -> float:
    glide_frames = [
        float(frame["midi"])
        for frame in _frame_window(frame_pitch_debug, start_time, end_time)
        if frame.get("voiced") and frame.get("midi") is not None
    ]
    if len(glide_frames) < 3 or abs(end_midi - start_midi) < 1:
        return 0.0
    smoothed = np.asarray(glide_frames, dtype=float)
    direction = np.sign(end_midi - start_midi)
    deltas = np.diff(smoothed)
    if deltas.size == 0:
        return 0.0
    if direction > 0:
        monotonicity = float(np.mean(deltas >= -0.4))
    else:
        monotonicity = float(np.mean(deltas <= 0.4))
    covered_range = (
        float(np.min(smoothed)) <= min(start_midi, end_midi) + 0.75
        and float(np.max(smoothed)) >= max(start_midi, end_midi) - 0.75
    )
    return monotonicity if covered_range else monotonicity * 0.5


def _segment_audio(audio: np.ndarray, sample_rate: int, start_time: float, end_time: float) -> np.ndarray:
    start_index = max(0, int(start_time * sample_rate))
    end_index = min(len(audio), max(start_index + 1, int(end_time * sample_rate)))
    return audio[start_index:end_index]


def _annotate_articulation_candidates(
    notes: list[MelodyNote],
    frame_pitch_debug: list[dict[str, object]],
    audio: np.ndarray,
    sample_rate: int,
    source_classification: SourceClassification,
) -> list[MelodyNote]:
    if not notes:
        return notes

    for note in notes:
        note.slide_to_next = False
        note.slide_from_previous = False
        note.harmonic_candidate = False
        note.articulation_hint = None

    ordered_notes = sorted(notes, key=lambda item: (item.start_time, item.note_index))

    for current, following in zip(ordered_notes, ordered_notes[1:]):
        same_string_pair = _best_same_string_pair(current.midi, following.midi)
        if same_string_pair is None:
            continue
        string_name, current_fret, following_fret = same_string_pair
        interval = following.midi - current.midi
        gap = following.start_time - current.end_time
        glide_score = _monotonic_glide_score(
            frame_pitch_debug,
            max(current.start_time, current.end_time - 0.06),
            min(following.end_time, following.start_time + 0.08),
            current.midi,
            following.midi,
        )
        if (
            source_classification.source_family == "ukulele"
            and 1 <= abs(interval) <= 5
            and abs(following_fret - current_fret) >= 1
            and gap <= 0.12
            and (
                glide_score >= 0.72
                or (
                    abs(following_fret - current_fret) <= 4
                    and (
                        not following.onset_supported
                        or following.onset_strength < 0.24
                        or current_fret >= 5
                        or following_fret >= 5
                    )
                )
            )
        ):
            current.slide_to_next = True
            following.slide_from_previous = True
            current.articulation_hint = "slide"
            following.articulation_hint = "slide"

    global_rms = float(np.sqrt(np.mean(np.square(audio)))) + 1e-6
    for index, note in enumerate(ordered_notes):
        segment = _segment_audio(audio, sample_rate, note.start_time, note.end_time)
        if segment.size < 256:
            continue
        centroid = _safe_mean(librosa.feature.spectral_centroid(y=segment, sr=sample_rate)[0])
        flatness = _safe_mean(librosa.feature.spectral_flatness(y=segment)[0])
        zcr = _safe_mean(librosa.feature.zero_crossing_rate(y=segment)[0])
        rms = float(np.sqrt(np.mean(np.square(segment)))) + 1e-6
        harmonic_segment, percussive_segment = librosa.effects.hpss(segment)
        harmonic_ratio = (float(np.sum(np.square(harmonic_segment))) + 1e-6) / (
            float(np.sum(np.square(percussive_segment))) + 1e-6
        )
        mapped_string, mapped_fret = _draft_pdf_mapping_for_note(note)
        harmonic_like = (
            source_classification.source_family == "ukulele"
            and index == len(ordered_notes) - 1
            and note.onset_strength >= 0.05
            and note.duration <= 1.0
            and centroid >= max(1300.0, note.frequency_hz * 2.5)
            and harmonic_ratio >= 1.25
            and flatness <= 0.14
            and zcr <= 0.16
            and rms <= global_rms * 1.35
            and (mapped_fret in {5, 7, 12} or note.midi >= 72 or centroid >= 2200.0)
        )
        if harmonic_like:
            note.harmonic_candidate = True
            note.articulation_hint = "harmonic"
            note.string_name = mapped_string
            note.fret_number = mapped_fret

    return notes


def _prepare_audio_source(audio_path: str | Path, working_dir: str | Path | None = None) -> Path:
    path = Path(audio_path)
    if path.suffix.lower() not in VIDEO_EXTENSIONS:
        return path

    target_dir = Path(working_dir) if working_dir else path.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    extracted_wav = target_dir / f"{path.stem}__extracted_audio.wav"
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())

    command = [
        str(ffmpeg_exe),
        "-y",
        "-i",
        str(path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "22050",
        "-ac",
        "1",
        str(extracted_wav),
    ]
    subprocess.run(
        command,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return extracted_wav


def _median_or_default(values: list[float] | np.ndarray, default: float) -> float:
    if len(values) == 0:
        return default
    return float(np.median(np.asarray(values, dtype=float)))


def _snap_to_sixteenth_grid(value: float) -> float:
    return round(value * 4) / 4


def _quantize_duration_to_grid(duration_beats: float) -> float:
    return max(0.25, _snap_to_sixteenth_grid(duration_beats))


def _measure_and_beat_from_position(
    quantized_beat_position: float,
    beats_per_measure: int,
) -> tuple[int, float]:
    snapped_position = round(float(quantized_beat_position), 4)
    measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
    beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)
    return measure_number, beat_in_measure


def _set_note_measure_and_beat(note: MelodyNote, beats_per_measure: int) -> None:
    measure_number, beat_in_measure = _measure_and_beat_from_position(
        note.quantized_beat_position,
        beats_per_measure,
    )
    note.measure_number = measure_number
    note.beat_in_measure = beat_in_measure
    note.beat_start_in_measure = beat_in_measure
    note.duration_beats = round(note.quantized_duration_beats, 4)


def _strong_beat_profile(beats_per_measure: int) -> dict[int, float]:
    if beats_per_measure == 4:
        return {0: 1.0, 1: 0.45, 2: 0.7, 3: 0.45}
    if beats_per_measure == 3:
        return {0: 1.0, 1: 0.48, 2: 0.48}
    return {index: 1.0 if index == 0 else 0.5 for index in range(beats_per_measure)}


def _dedupe_beat_times(
    beat_times: list[float],
    min_spacing_seconds: float = 0.2,
) -> list[float]:
    if not beat_times:
        return []

    deduped: list[float] = []
    for time_point in sorted(float(value) for value in beat_times):
        if deduped and (time_point - deduped[-1]) < min_spacing_seconds:
            if time_point > deduped[-1]:
                deduped[-1] = round((deduped[-1] + time_point) / 2.0, 4)
            continue
        deduped.append(round(time_point, 4))
    return deduped


def _estimate_bpm_from_beat_times(beat_times: list[float]) -> float:
    if len(beat_times) < 2:
        return 0.0

    diffs = np.diff(np.asarray(beat_times, dtype=float))
    plausible_diffs = diffs[(diffs >= 0.18) & (diffs <= 2.0)]
    if plausible_diffs.size == 0:
        return 0.0

    beat_duration = float(np.median(plausible_diffs))
    if beat_duration <= 0:
        return 0.0
    return 60.0 / beat_duration


def _uniform_beat_times(
    duration_seconds: float,
    bpm_estimate: float,
) -> list[float]:
    if bpm_estimate <= 0:
        bpm_estimate = 90.0
    beat_duration = 60.0 / bpm_estimate
    beat_count = max(2, int(np.ceil(duration_seconds / beat_duration)) + 2)
    return [round(index * beat_duration, 4) for index in range(beat_count)]


def _extract_librosa_beat_track_candidate(
    onset_env: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
) -> tuple[str, list[float], float, str]:
    tempo, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=BEAT_HOP_LENGTH,
        trim=False,
    )
    beat_times = librosa.frames_to_time(
        beat_frames,
        sr=sample_rate,
        hop_length=BEAT_HOP_LENGTH,
    ).tolist()
    beat_times = _dedupe_beat_times(beat_times, min_spacing_seconds=0.18)
    bpm_estimate = _estimate_bpm_from_beat_times(beat_times)
    if bpm_estimate <= 0:
        bpm_estimate = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if bpm_estimate > 0 and not beat_times:
        beat_times = _uniform_beat_times(duration_seconds, bpm_estimate)
    return (
        "librosa.beat_track",
        beat_times,
        bpm_estimate,
        "Global beat tracking from librosa onset envelope.",
    )


def _extract_plp_beat_candidate(
    onset_env: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
) -> tuple[str, list[float], float, str] | None:
    pulse = librosa.beat.plp(
        onset_envelope=onset_env,
        sr=sample_rate,
        hop_length=BEAT_HOP_LENGTH,
    )
    if pulse.size == 0:
        return None

    peak_mask = librosa.util.localmax(pulse)
    threshold = max(float(np.percentile(pulse, 75)), float(np.max(pulse)) * 0.35)
    peak_frames = np.flatnonzero(peak_mask & (pulse >= threshold))
    beat_times = librosa.frames_to_time(
        peak_frames,
        sr=sample_rate,
        hop_length=BEAT_HOP_LENGTH,
    ).tolist()
    beat_times = _dedupe_beat_times(beat_times, min_spacing_seconds=0.22)
    bpm_estimate = _estimate_bpm_from_beat_times(beat_times)
    if bpm_estimate <= 0 or len(beat_times) < 2:
        return None

    if beat_times[0] > 0.28:
        beat_times = [0.0, *beat_times]
    if beat_times[-1] < max(0.0, duration_seconds - 0.32):
        beat_times.append(round(beat_times[-1] + (60.0 / bpm_estimate), 4))
    return (
        "librosa.plp",
        beat_times,
        bpm_estimate,
        "Predominant local pulse from librosa, peak-picked into beat candidates.",
    )


def _extract_aubio_beat_candidate(
    audio: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
) -> tuple[str, list[float], float, str] | None:
    if aubio is None:
        return None

    try:
        win_size = 1024
        hop_size = BEAT_HOP_LENGTH
        tempo_detector = aubio.tempo("default", win_size, hop_size, sample_rate)
        beat_times: list[float] = []
        mono_audio = np.asarray(audio, dtype=np.float32)
        for start in range(0, len(mono_audio), hop_size):
            frame = mono_audio[start : start + hop_size]
            if len(frame) < hop_size:
                frame = np.pad(frame, (0, hop_size - len(frame)))
            if tempo_detector(frame):
                beat_times.append(float(tempo_detector.get_last_s()))
        beat_times = _dedupe_beat_times(beat_times, min_spacing_seconds=0.18)
        bpm_estimate = _estimate_bpm_from_beat_times(beat_times)
        if bpm_estimate <= 0:
            detected_bpm = float(tempo_detector.get_bpm())
            bpm_estimate = detected_bpm if detected_bpm > 0 else 0.0
        if bpm_estimate > 0 and not beat_times:
            beat_times = _uniform_beat_times(duration_seconds, bpm_estimate)
        if bpm_estimate <= 0 or len(beat_times) < 2:
            return None
        return (
            "aubio.tempo",
            beat_times,
            bpm_estimate,
            "Beat events from aubio tempo detector.",
        )
    except Exception:
        return None


def _beat_regularization_score(beat_times: list[float], duration_seconds: float) -> float:
    if len(beat_times) < 2:
        return float("-inf")

    diffs = np.diff(np.asarray(beat_times, dtype=float))
    plausible_diffs = diffs[(diffs >= 0.18) & (diffs <= 2.0)]
    if plausible_diffs.size == 0:
        return float("-inf")

    median_diff = float(np.median(plausible_diffs))
    mean_abs_error = float(np.mean(np.abs(plausible_diffs - median_diff)))
    coverage = min(1.0, (len(beat_times) * median_diff) / max(duration_seconds, 1e-6))
    return -mean_abs_error * 2.4 + coverage * 0.22


def _collect_beat_candidates(
    audio: np.ndarray,
    onset_env: np.ndarray,
    sample_rate: int,
    duration_seconds: float,
) -> list[tuple[str, list[float], float, str]]:
    candidates: list[tuple[str, list[float], float, str]] = []
    for candidate in (
        _extract_librosa_beat_track_candidate(onset_env, sample_rate, duration_seconds),
        _extract_plp_beat_candidate(onset_env, sample_rate, duration_seconds),
        _extract_aubio_beat_candidate(audio, sample_rate, duration_seconds),
    ):
        if candidate is None:
            continue
        source_name, beat_times, bpm_estimate, note = candidate
        if bpm_estimate <= 0 or len(beat_times) < 2:
            continue
        candidates.append((source_name, beat_times, bpm_estimate, note))

    if not candidates:
        fallback_bpm = 90.0
        candidates.append(
            (
                "uniform_fallback",
                _uniform_beat_times(duration_seconds, fallback_bpm),
                fallback_bpm,
                "Fallback uniform beat grid because no beat backend produced enough events.",
            )
        )
    return candidates


def _rhythm_value_from_duration(duration_beats: float) -> str:
    if duration_beats <= 0.25:
        return "sixteenth"
    if duration_beats <= 0.5:
        return "eighth"
    if duration_beats <= 0.75:
        return "dotted_eighth"
    if duration_beats <= 1.0:
        return "quarter"
    if duration_beats <= 2.0:
        return "half"
    return "whole"


def _classify_rhythm_value(
    duration_beats: float,
    previous_step: float | None,
    next_step: float | None,
) -> str:
    local_steps = [step for step in (previous_step, next_step) if step is not None]
    min_step = min(local_steps) if local_steps else None
    short_step_count = sum(step <= 0.3 for step in local_steps)
    medium_step_count = sum(step <= 0.55 for step in local_steps)

    if duration_beats >= 3.25:
        return "whole"
    if duration_beats >= 2.75:
        return "dotted_half"
    if duration_beats >= 1.75:
        return "half"
    if duration_beats >= 1.25:
        return "dotted_quarter"
    if duration_beats >= 0.875:
        return "quarter"
    if duration_beats >= 0.625:
        return "dotted_eighth"
    if duration_beats >= 0.375:
        return "eighth"

    # Sixteenth notes need explicit evidence that the note sits inside
    # a denser beat subdivision instead of being a short but isolated pluck.
    if (
        duration_beats <= 0.25
        and (
            short_step_count >= 2
            or (
                short_step_count >= 1
                and medium_step_count >= 2
                and min_step is not None
                and min_step <= 0.26
            )
        )
    ):
        return "sixteenth"
    if duration_beats <= 0.5:
        return "eighth"
    return "quarter"


def _beam_level_from_rhythm_value(rhythm_value: str) -> int:
    if rhythm_value == "sixteenth":
        return 2
    if rhythm_value in {"eighth", "dotted_eighth", "triplet_eighth"}:
        return 1
    return 0


def _match_offsets(
    offsets: list[float],
    expected: list[float],
    tolerance: float,
) -> bool:
    if len(offsets) != len(expected):
        return False
    return all(abs(actual - target) <= tolerance for actual, target in zip(offsets, expected))


def _supports_triplet_interpretation(
    beat_notes: list[MelodyNote],
    offsets: list[float],
) -> bool:
    if len(beat_notes) != 3:
        return False
    if not _match_offsets(offsets, [0.0, 1 / 3, 2 / 3], tolerance=0.14):
        return False

    same_pitch = len({note.midi for note in beat_notes}) == 1
    durations = [max(note.duration, 0.001) for note in beat_notes]
    max_duration = max(durations)
    min_duration = min(durations)
    duration_ratio = max_duration / min_duration if min_duration > 0 else float("inf")
    average_duration = float(np.mean(durations))
    duration_spread = max(abs(duration - average_duration) for duration in durations)

    if same_pitch:
        return True

    # For mixed-pitch groups, only accept triplets when durations are close enough
    # to a true equal subdivision. This avoids rewriting short-pickup-plus-anchor
    # shapes into triplets.
    return duration_ratio <= 1.45 and duration_spread <= average_duration * 0.32


def _recognize_beat_pattern(beat_notes: list[MelodyNote]) -> str:
    if not beat_notes:
        return "empty"

    if len(beat_notes) == 1:
        duration_beats = beat_notes[0].quantized_duration_beats
        if duration_beats >= 2.75:
            return "dotted_half"
        if duration_beats >= 1.75:
            return "half"
        if duration_beats >= 1.25:
            return "dotted_quarter"
        return "quarter"

    quantized_offsets = [
        round((note.beat_in_measure - 1) - np.floor(note.beat_in_measure - 1), 4)
        for note in beat_notes
    ]
    raw_offsets = [
        round(note.raw_beat_position - np.floor(note.raw_beat_position), 4)
        for note in beat_notes
    ]

    if len(beat_notes) == 3 and _supports_triplet_interpretation(beat_notes, raw_offsets):
        return "triplet_eighths"
    if _match_offsets(quantized_offsets, [0.0, 0.25], tolerance=0.08):
        return "sixteenth_then_dotted_eighth"
    if _match_offsets(quantized_offsets, [0.0, 0.5], tolerance=0.08):
        return "two_eighths"
    if _match_offsets(quantized_offsets, [0.0, 0.75], tolerance=0.08):
        return "dotted_eighth_then_sixteenth"
    if _match_offsets(quantized_offsets, [0.0, 0.25, 0.5], tolerance=0.08):
        return "front_sixteenth_then_eighth"
    if _match_offsets(quantized_offsets, [0.0, 0.5, 0.75], tolerance=0.08):
        return "front_eighth_then_sixteenth"
    if _match_offsets(quantized_offsets, [0.0, 0.25, 0.5, 0.75], tolerance=0.08):
        return "four_sixteenths"

    # Fallbacks for imperfect segmentation: prefer recognizable beat-level shapes.
    if len(beat_notes) == 2:
        durations = [note.duration for note in beat_notes]
        raw_gap = round(raw_offsets[1] - raw_offsets[0], 4)
        if durations[0] >= max(durations[1] * 1.45, 0.15) and raw_gap >= 0.58:
            return "dotted_eighth_then_sixteenth"
        if durations[1] >= max(durations[0] * 1.45, 0.15) and raw_gap <= 0.36:
            return "sixteenth_then_dotted_eighth"
        return "two_eighths"
    if len(beat_notes) == 3:
        durations = [note.duration for note in beat_notes]
        if durations[0] < min(durations[1], durations[2]) * 0.72:
            return "front_sixteenth_then_eighth"
        if durations[2] < min(durations[0], durations[1]) * 0.72:
            return "front_eighth_then_sixteenth"
        first_gap = round(quantized_offsets[1] - quantized_offsets[0], 4)
        second_gap = round(quantized_offsets[2] - quantized_offsets[1], 4)
        if first_gap <= 0.3 and second_gap <= 0.3:
            return "front_sixteenth_then_eighth"
        if first_gap >= 0.45 and second_gap <= 0.3:
            return "front_eighth_then_sixteenth"
        return "front_sixteenth_then_eighth" if first_gap <= second_gap else "front_eighth_then_sixteenth"
    if len(beat_notes) >= 4:
        return "four_sixteenths"

    return "quarter"


def _apply_beat_pattern(
    beat_notes: list[MelodyNote],
    pattern_name: str,
    beam_counter: int,
    tuplet_counter: int,
) -> tuple[int, int]:
    if not beat_notes:
        return beam_counter, tuplet_counter

    beam_id = None
    tuplet_id = None
    if pattern_name in {
        "two_eighths",
        "dotted_eighth_then_sixteenth",
        "sixteenth_then_dotted_eighth",
        "front_sixteenth_then_eighth",
        "front_eighth_then_sixteenth",
        "four_sixteenths",
        "triplet_eighths",
    } and len(beat_notes) >= 2:
        beam_id = f"beam_{beam_counter}"
        beam_counter += 1
    if pattern_name == "triplet_eighths":
        tuplet_id = f"triplet_{tuplet_counter}"
        tuplet_counter += 1

    for index, note in enumerate(beat_notes):
        note.beat_pattern = pattern_name
        note.beam_group = beam_id
        note.tuplet_group = tuplet_id

        if pattern_name == "quarter":
            note.rhythm_value = "quarter"
            note.quantized_duration_beats = max(1.0, note.quantized_duration_beats)
            continue
        if pattern_name == "dotted_quarter":
            note.rhythm_value = "dotted_quarter"
            note.quantized_duration_beats = max(1.5, note.quantized_duration_beats)
            continue
        if pattern_name == "half":
            note.rhythm_value = "half"
            note.quantized_duration_beats = max(2.0, note.quantized_duration_beats)
            continue
        if pattern_name == "dotted_half":
            note.rhythm_value = "dotted_half"
            note.quantized_duration_beats = max(3.0, note.quantized_duration_beats)
            continue
        if pattern_name == "two_eighths":
            note.rhythm_value = "eighth"
            note.quantized_duration_beats = 0.5
            continue
        if pattern_name == "dotted_eighth_then_sixteenth":
            note.rhythm_value = "dotted_eighth" if index == 0 else "sixteenth"
            note.quantized_duration_beats = 0.75 if index == 0 else 0.25
            continue
        if pattern_name == "sixteenth_then_dotted_eighth":
            note.rhythm_value = "sixteenth" if index == 0 else "dotted_eighth"
            note.quantized_duration_beats = 0.25 if index == 0 else 0.75
            continue
        if pattern_name == "front_sixteenth_then_eighth":
            note.rhythm_value = "sixteenth" if index < 2 else "eighth"
            note.quantized_duration_beats = 0.25 if index < 2 else 0.5
            continue
        if pattern_name == "front_eighth_then_sixteenth":
            note.rhythm_value = "eighth" if index == 0 else "sixteenth"
            note.quantized_duration_beats = 0.5 if index == 0 else 0.25
            continue
        if pattern_name == "four_sixteenths":
            note.rhythm_value = "sixteenth"
            note.quantized_duration_beats = 0.25
            continue
        if pattern_name == "triplet_eighths":
            note.rhythm_value = "triplet_eighth"
            note.quantized_duration_beats = round(1 / 3, 4)

    return beam_counter, tuplet_counter


def _build_notation_analysis(
    notes: list[MelodyNote],
    rhythm: RhythmAnalysis,
) -> TabNotationAnalysis:
    measure_numbers = sorted({note.measure_number for note in notes if note.measure_number > 0})
    measures_per_system = 4
    systems: list[TabSystemLayout] = []
    for system_index, start in enumerate(range(0, len(measure_numbers), measures_per_system), start=1):
        systems.append(
            TabSystemLayout(
                system_index=system_index,
                measure_numbers=measure_numbers[start:start + measures_per_system],
                string_count=4,
                string_labels=PDF_STRING_ORDER.copy(),
            )
        )

    symbol_events: list[TabSymbolEvent] = []
    for note in notes:
        string_name, fret_number = _draft_pdf_mapping_for_note(note)
        note.string_name = string_name
        note.fret_number = fret_number
        note.beat_start_in_measure = round(note.beat_in_measure, 4)
        note.duration_beats = round(note.quantized_duration_beats, 4)
        symbol_events.append(
            TabSymbolEvent(
                note_index=note.note_index,
                measure_number=note.measure_number,
                string_name=string_name,
                fret_number=fret_number,
                beat_start_in_measure=round(note.beat_in_measure, 4),
                duration_beats=round(note.quantized_duration_beats, 4),
                rhythm_value=note.rhythm_value,
                beam_group=note.beam_group,
                beam_level=_beam_level_from_rhythm_value(note.rhythm_value),
                beat_pattern=note.beat_pattern,
                tuplet_group=note.tuplet_group,
                slide_to_next=note.slide_to_next,
                slide_from_previous=note.slide_from_previous,
                harmonic_candidate=note.harmonic_candidate,
                articulation_hint=note.articulation_hint,
            )
        )

    return TabNotationAnalysis(
        page_layout=TabPageLayout(
            system_count=len(systems),
            systems=systems,
            beats_per_measure=rhythm.beats_per_measure,
            bpm_estimate=rhythm.bpm_estimate,
            time_signature=f"{rhythm.beats_per_measure}/4",
        ),
        symbol_events=symbol_events,
        notes=(
            "This notation layer separates page layout, symbol events, and rhythmic values. "
            "It is intended to support future score-image comparison and more faithful PDF rendering."
        ),
    )


def _assign_beam_group(
    notes: list[MelodyNote],
    beam_counter: int,
) -> int:
    if len(notes) < 2:
        return beam_counter

    short_notes = [note for note in notes if _beam_level_from_rhythm_value(note.rhythm_value) > 0]
    if len(short_notes) < 2:
        return beam_counter

    beam_id = f"beam_{beam_counter}"
    for note in short_notes:
        note.beam_group = beam_id
    return beam_counter + 1


def preprocess_audio(
    audio_path: str | Path,
    target_sr: int = 22050,
    working_dir: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, AudioPreprocessSummary]:
    source_path = Path(audio_path)
    path = _prepare_audio_source(source_path, working_dir=working_dir)
    full_audio, sample_rate = librosa.load(path, sr=target_sr, mono=True)
    trimmed_audio, trim_index = librosa.effects.trim(full_audio, top_db=30)
    if trimmed_audio.size == 0:
        trimmed_audio = full_audio
        trim_index = np.array([0, len(full_audio)])

    peak = float(np.max(np.abs(trimmed_audio))) if trimmed_audio.size else 0.0
    normalized_audio = trimmed_audio / peak if peak > 0 else trimmed_audio
    normalized_full_audio = full_audio / peak if peak > 0 else full_audio

    summary = AudioPreprocessSummary(
        source_path=str(source_path),
        sample_rate=target_sr,
        channels=1,
        trimmed_start_seconds=round(trim_index[0] / target_sr, 4),
        trimmed_end_seconds=round((len(full_audio) - trim_index[1]) / target_sr, 4),
        duration_seconds=round(float(librosa.get_duration(y=normalized_audio, sr=target_sr)), 4),
        normalized_peak=round(float(np.max(np.abs(normalized_audio))) if normalized_audio.size else 0.0, 4),
    )
    return normalized_audio, normalized_full_audio, summary


def _estimate_beats(
    audio: np.ndarray,
    sample_rate: int,
    time_offset_seconds: float = 0.0,
    visible_duration_seconds: float | None = None,
) -> tuple[RhythmAnalysis, np.ndarray]:
    onset_env = librosa.onset.onset_strength(y=audio, sr=sample_rate)
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sample_rate, trim=False)
    beat_times_full = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
    bpm_estimate = float(np.atleast_1d(tempo)[0]) if np.size(tempo) else 0.0
    if bpm_estimate <= 0:
        bpm_estimate = 90.0

    duration_seconds = float(librosa.get_duration(y=audio, sr=sample_rate))
    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0

    if not beat_times_full:
        beat_times_full = [
            round(index * beat_duration, 4)
            for index in range(max(1, int(np.ceil(duration_seconds / beat_duration))))
        ]
        beat_frames = librosa.time_to_frames(np.asarray(beat_times_full), sr=sample_rate)

    beat_frames = np.asarray(beat_frames, dtype=int)
    beat_strengths: list[float] = []
    for frame in beat_frames:
        clamped = int(np.clip(frame, 0, max(0, len(onset_env) - 1)))
        beat_strengths.append(float(onset_env[clamped]) if onset_env.size else 0.0)

    def meter_grid_score(beats_per_measure: int, phase: int) -> float:
        if not beat_times_full:
            return float("-inf")

        first_measure_start = float(beat_times_full[phase]) - phase * beat_duration
        visible_beats = [
            time_point - time_offset_seconds
            for time_point in beat_times_full
            if -beat_duration <= (time_point - time_offset_seconds) <= visible_duration_seconds + beat_duration
        ]
        if len(visible_beats) < beats_per_measure:
            return float("-inf")

        note_start_frames = np.where(onset_env >= np.percentile(onset_env, 80) if onset_env.size else 0.0)[0]
        note_start_times = librosa.frames_to_time(note_start_frames, sr=sample_rate).tolist()
        if not note_start_times:
            note_start_times = beat_times_full
        note_start_strengths = [
            float(onset_env[int(np.clip(frame, 0, max(0, len(onset_env) - 1)))]) if onset_env.size else 0.0
            for frame in note_start_frames
        ]
        if not note_start_strengths:
            note_start_strengths = [1.0 for _ in note_start_times]

        grid_errors: list[float] = []
        repetition_bonus = 0.0
        repeating_ioi_count = 0
        quantized_positions: list[float] = []
        accent_alignment = 0.0
        offbeat_penalty = 0.0
        barline_alignment_bonus = 0.0
        previous_note_start: float | None = None
        previous_ioi_beats: float | None = None
        accent_profile = _strong_beat_profile(beats_per_measure)

        for note_start_full, note_strength in zip(note_start_times, note_start_strengths):
            relative = (note_start_full - first_measure_start) / beat_duration
            snapped = _snap_to_sixteenth_grid(relative)
            quantized_positions.append(snapped)
            grid_errors.append(abs(relative - snapped))
            measure_fraction = relative % beats_per_measure
            nearest_primary = round(measure_fraction)
            primary_error = abs(measure_fraction - nearest_primary)
            if primary_error <= 0.16:
                beat_index = int(nearest_primary) % beats_per_measure
                accent_alignment += note_strength * accent_profile.get(beat_index, 0.5)
                if beat_index == 0:
                    barline_alignment_bonus += 0.015
            elif primary_error >= 0.3:
                offbeat_penalty += note_strength * 0.08
            if previous_note_start is not None:
                ioi_beats = (note_start_full - previous_note_start) / beat_duration
                snapped_ioi = _quantize_duration_to_grid(ioi_beats)
                if abs(ioi_beats - snapped_ioi) <= 0.18:
                    repetition_bonus += 0.03
                if previous_ioi_beats is not None and abs(snapped_ioi - previous_ioi_beats) <= 0.01:
                    repeating_ioi_count += 1
                previous_ioi_beats = snapped_ioi
            previous_note_start = note_start_full

        measure_position_bonus = 0.0
        if quantized_positions:
            downbeats = sum(abs(position % beats_per_measure) <= 0.08 for position in quantized_positions)
            measure_position_bonus = downbeats * 0.01

        average_grid_error = float(np.mean(grid_errors)) if grid_errors else 1.0
        four_four_bias = 0.03 if beats_per_measure == 4 else 0.0
        return (
            -average_grid_error * 3.0
            + repetition_bonus
            + repeating_ioi_count * 0.015
            + measure_position_bonus
            + accent_alignment * 0.08
            + barline_alignment_bonus
            - offbeat_penalty
            + four_four_bias
        )

    meter_candidates = [3, 4]
    best_meter = 4
    best_phase = 0
    best_score = float("-inf")
    for beats_per_measure in meter_candidates:
        for phase in range(beats_per_measure):
            strong = [strength for index, strength in enumerate(beat_strengths) if index % beats_per_measure == phase]
            weak = [strength for index, strength in enumerate(beat_strengths) if index % beats_per_measure != phase]
            if not strong:
                continue
            strong_mean = float(np.mean(strong))
            weak_mean = float(np.mean(weak)) if weak else 0.0
            accent_score = strong_mean - weak_mean + len(strong) * 0.001
            grid_score = meter_grid_score(beats_per_measure, phase)
            score = accent_score * 0.35 + grid_score
            if score > best_score:
                best_score = score
                best_meter = beats_per_measure
                best_phase = phase

    beats_per_measure = best_meter
    first_measure_start_full = float(beat_times_full[best_phase]) - best_phase * beat_duration if beat_times_full else 0.0
    grid_origin_seconds = round(first_measure_start_full - time_offset_seconds, 4)

    beat_times = [round(time_point - time_offset_seconds, 4) for time_point in beat_times_full]
    if visible_duration_seconds is None:
        visible_duration_seconds = max(0.0, duration_seconds - time_offset_seconds)
    visible_duration_seconds = float(visible_duration_seconds)

    strong_beats = []
    measure_boundaries = []
    current_boundary = grid_origin_seconds
    while current_boundary + beat_duration * beats_per_measure < 0:
        current_boundary += beat_duration * beats_per_measure
    while current_boundary <= visible_duration_seconds + beat_duration:
        rounded_boundary = round(current_boundary, 4)
        measure_boundaries.append(rounded_boundary)
        strong_beats.append(rounded_boundary)
        current_boundary += beat_duration * beats_per_measure

    if not measure_boundaries:
        measure_boundaries = [0.0]
        strong_beats = [0.0]

    rhythm = RhythmAnalysis(
        bpm_estimate=round(bpm_estimate, 2),
        bpm_confidence_note=(
            "Estimated from onset strength, beat tracking, and meter accent scoring; "
            "verify against the input melody and strong-beat placement."
        ),
        beat_positions=[
            round(value, 4)
            for value in beat_times
            if -beat_duration <= value <= visible_duration_seconds + beat_duration
        ],
        strong_beat_positions=strong_beats,
        measure_boundaries=measure_boundaries,
        beats_per_measure=beats_per_measure,
        grid_origin_seconds=grid_origin_seconds,
        quantization_grid="1/16 note grid",
        time_signature_tendency=f"{beats_per_measure}/4 tendency",
    )
    return rhythm, np.array(beat_times, dtype=float)


def _build_pitch_histogram(midi_values: list[int]) -> list[tuple[str, int]]:
    counts: dict[str, int] = {}
    for midi_note in midi_values:
        pitch_class = PITCH_CLASSES[midi_note % 12]
        counts[pitch_class] = counts.get(pitch_class, 0) + 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def _extract_pitch_frames(
    audio: np.ndarray,
    sample_rate: int,
    hop_length: int = HOP_LENGTH,
) -> tuple[list[dict[str, object]], np.ndarray]:
    harmonic_audio = librosa.effects.harmonic(audio, margin=4.0)
    percussive_audio = librosa.effects.percussive(audio, margin=4.0)
    onset_envelope = librosa.onset.onset_strength(y=audio, sr=sample_rate, hop_length=hop_length)
    percussive_onset_envelope = librosa.onset.onset_strength(y=percussive_audio, sr=sample_rate, hop_length=hop_length)
    onset_frames_primary = librosa.onset.onset_detect(
        onset_envelope=onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
    )
    onset_frames_percussive = librosa.onset.onset_detect(
        onset_envelope=percussive_onset_envelope,
        sr=sample_rate,
        hop_length=hop_length,
        units="frames",
        backtrack=False,
        pre_max=1,
        post_max=1,
        pre_avg=2,
        post_avg=2,
        wait=2,
    )
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)[0]
    raw_onset_frames = np.unique(np.concatenate([onset_frames_primary, onset_frames_percussive])).astype(int)
    onset_frames_backtracked = librosa.onset.onset_backtrack(
        raw_onset_frames,
        rms,
    )
    onset_frames = np.unique(np.concatenate([raw_onset_frames, onset_frames_backtracked])).astype(int)
    f0, voiced_flags, voiced_probabilities = librosa.pyin(
        harmonic_audio,
        fmin=librosa.note_to_hz(UKULELE_NOTE_MIN),
        fmax=librosa.note_to_hz(UKULELE_NOTE_MAX),
        sr=sample_rate,
        frame_length=2048,
        hop_length=hop_length,
    )
    frame_times = librosa.times_like(f0, sr=sample_rate, hop_length=hop_length)
    frame_pitch_debug: list[dict[str, object]] = []
    onset_frame_set = {int(value) for value in np.atleast_1d(onset_frames).tolist()}
    primary_onset_set = {int(value) for value in np.atleast_1d(onset_frames_primary).tolist()}
    percussive_onset_set = {int(value) for value in np.atleast_1d(onset_frames_percussive).tolist()}
    if rms.size:
        rms_norm = rms / max(float(np.max(rms)), 1e-6)
    else:
        rms_norm = rms
    if onset_envelope.size:
        onset_norm = onset_envelope / max(float(np.max(onset_envelope)), 1e-6)
    else:
        onset_norm = onset_envelope
    if percussive_onset_envelope.size:
        percussive_norm = percussive_onset_envelope / max(float(np.max(percussive_onset_envelope)), 1e-6)
    else:
        percussive_norm = percussive_onset_envelope

    for index, time_point in enumerate(frame_times):
        frequency = f0[index]
        voiced = bool(voiced_flags[index]) if index < len(voiced_flags) else False
        probability = float(voiced_probabilities[index]) if index < len(voiced_probabilities) and not np.isnan(voiced_probabilities[index]) else 0.0
        onset_strength = float(
            max(
                onset_norm[index] if index < len(onset_norm) else 0.0,
                percussive_norm[index] if index < len(percussive_norm) else 0.0,
            )
        )
        onset_peak = index in primary_onset_set or index in percussive_onset_set
        current_rms = float(rms_norm[index]) if index < len(rms_norm) else 0.0
        next_rms = float(rms_norm[index + 1]) if index + 1 < len(rms_norm) else 0.0
        offset_strength = max(0.0, current_rms - next_rms)
        offset_peak = offset_strength >= 0.12
        if np.isnan(frequency) or not voiced:
            frame_pitch_debug.append(
                {
                    "frame_index": index,
                    "time": round(float(time_point), 4),
                    "frequency_hz": None,
                    "midi": None,
                    "note_name": None,
                    "confidence": round(probability, 4),
                    "voiced": False,
                    "onset_frame": index in onset_frame_set,
                    "onset_peak": onset_peak,
                    "onset_strength": round(onset_strength, 4),
                    "offset_peak": offset_peak,
                    "offset_strength": round(offset_strength, 4),
                    "energy": round(current_rms, 4),
                }
            )
            continue

        midi_note = int(round(librosa.hz_to_midi(float(frequency))))
        frame_pitch_debug.append(
            {
                "frame_index": index,
                "time": round(float(time_point), 4),
                "frequency_hz": round(float(frequency), 3),
                "midi": midi_note,
                "note_name": _format_note_name(midi_note),
                "confidence": round(probability, 4),
                "voiced": True,
                "onset_frame": index in onset_frame_set,
                "onset_peak": onset_peak,
                "onset_strength": round(onset_strength, 4),
                "offset_peak": offset_peak,
                "offset_strength": round(offset_strength, 4),
                "energy": round(current_rms, 4),
            }
        )

    return frame_pitch_debug, onset_frames


def _split_segment_by_pitch_continuity(
    segment_frames: list[dict[str, object]],
    pitch_jump_threshold: int = 2,
    max_unvoiced_gap_frames: int = 2,
    min_run_frames: int = 2,
) -> list[list[dict[str, object]]]:
    voiced_indices = [index for index, frame in enumerate(segment_frames) if bool(frame["voiced"]) and frame["midi"] is not None]
    if not voiced_indices:
        return []

    subsegments: list[list[dict[str, object]]] = []
    current_start = voiced_indices[0]
    anchor_midis: list[int] = [int(segment_frames[current_start]["midi"])]
    pending_pitch_break_index: int | None = None
    pending_count = 0

    for local_index in voiced_indices[1:]:
        previous_voiced_index = max(index for index in voiced_indices if index < local_index)
        if local_index - previous_voiced_index > max_unvoiced_gap_frames:
            subsegments.append(segment_frames[current_start:previous_voiced_index + 1])
            current_start = local_index
            anchor_midis = [int(segment_frames[current_start]["midi"])]
            pending_pitch_break_index = None
            pending_count = 0
            continue

        current_frame = segment_frames[local_index]
        previous_frame = segment_frames[previous_voiced_index]
        internal_onset_break = (
            bool(current_frame.get("onset_peak"))
            and float(current_frame.get("onset_strength", 0.0)) >= 0.28
            and local_index - current_start >= 1
            and voiced_indices[-1] - local_index >= 1
        )
        repeated_pitch_reattack_break = (
            bool(current_frame.get("onset_peak"))
            and float(current_frame.get("onset_strength", 0.0)) >= 0.24
            and bool(previous_frame.get("offset_peak"))
            and float(previous_frame.get("offset_strength", 0.0)) >= 0.1
            and current_frame.get("midi") is not None
            and previous_frame.get("midi") is not None
            and abs(int(current_frame["midi"]) - int(previous_frame["midi"])) <= 1
            and float(current_frame.get("energy", 0.0)) >= float(previous_frame.get("energy", 0.0)) * 0.82
            and local_index - current_start >= 1
            and voiced_indices[-1] - local_index >= 1
        )
        if internal_onset_break or repeated_pitch_reattack_break:
            subsegments.append(segment_frames[current_start:local_index])
            current_start = local_index
            anchor_midis = [int(segment_frames[current_start]["midi"])]
            pending_pitch_break_index = None
            pending_count = 0
            continue

        current_midi = int(segment_frames[local_index]["midi"])
        anchor_midi = int(round(np.median(anchor_midis)))
        if abs(current_midi - anchor_midi) >= pitch_jump_threshold:
            if pending_pitch_break_index is None:
                pending_pitch_break_index = local_index
                pending_count = 1
            else:
                pending_count += 1

            if pending_count >= min_run_frames:
                split_end = max(current_start + 1, pending_pitch_break_index)
                subsegments.append(segment_frames[current_start:split_end])
                current_start = pending_pitch_break_index
                anchor_midis = [int(frame["midi"]) for frame in segment_frames[current_start:local_index + 1] if frame["midi"] is not None]
                pending_pitch_break_index = None
                pending_count = 0
            continue

        pending_pitch_break_index = None
        pending_count = 0
        anchor_midis.append(current_midi)

    if pending_pitch_break_index is not None:
        anchor_midi = int(round(np.median(anchor_midis))) if anchor_midis else int(segment_frames[current_start]["midi"])
        tail_frames = segment_frames[pending_pitch_break_index:voiced_indices[-1] + 1]
        tail_voiced = [
            frame for frame in tail_frames
            if bool(frame.get("voiced")) and frame.get("midi") is not None
        ]
        if tail_voiced:
            tail_first = tail_voiced[0]
            tail_pitch_shift = abs(int(tail_first["midi"]) - anchor_midi) >= pitch_jump_threshold
            tail_onset_supported = (
                bool(tail_first.get("onset_peak"))
                and float(tail_first.get("onset_strength", 0.0)) >= 0.2
            )
            tail_is_short = len(tail_voiced) <= 2
            if tail_is_short and (tail_pitch_shift or tail_onset_supported):
                split_end = max(current_start + 1, pending_pitch_break_index)
                subsegments.append(segment_frames[current_start:split_end])
                current_start = pending_pitch_break_index

    subsegments.append(segment_frames[current_start:voiced_indices[-1] + 1])
    return [segment for segment in subsegments if any(bool(frame["voiced"]) for frame in segment)]


def _segment_monophonic_notes(
    audio: np.ndarray,
    frame_pitch_debug: list[dict[str, object]],
    onset_frames: np.ndarray,
    sample_rate: int,
    allowed_midis: list[int] | None = None,
) -> list[MelodyNote]:
    hop_length = HOP_LENGTH
    notes: list[MelodyNote] = []
    frame_count = len(frame_pitch_debug)
    onset_list = sorted({0, *[int(value) for value in np.atleast_1d(onset_frames).tolist()]})
    if onset_list[-1] != frame_count:
        onset_list.append(frame_count)

    candidate_pool = allowed_midis if allowed_midis else UKULELE_PLAYABLE_MIDI

    def estimate_segment_pitch(
        segment_start_time: float,
        segment_end_time: float,
        fallback_midi: int,
        attack_focus: bool = False,
    ) -> tuple[int, float]:
        start_sample = max(0, int(segment_start_time * sample_rate))
        end_sample = min(len(audio), int(segment_end_time * sample_rate))
        segment_audio = audio[start_sample:end_sample]
        if segment_audio.size < 256:
            return fallback_midi, float(librosa.midi_to_hz(fallback_midi))

        short_segment = len(segment_audio) <= int(sample_rate * 0.2)
        if attack_focus:
            stable_start = 0
            stable_end = int(min(len(segment_audio), sample_rate * 0.09))
        else:
            stable_start = 0 if short_segment else int(min(len(segment_audio) * 0.25, sample_rate * 0.03))
            stable_end = int(min(len(segment_audio), stable_start + sample_rate * 0.22))
        analysis_audio = segment_audio[stable_start:stable_end] if stable_end > stable_start else segment_audio
        if analysis_audio.size < 256:
            analysis_audio = segment_audio

        windowed = analysis_audio * np.hanning(len(analysis_audio))
        spectrum = np.abs(np.fft.rfft(windowed))
        frequencies = np.fft.rfftfreq(len(windowed), d=1 / sample_rate)
        if spectrum.size == 0:
            return fallback_midi, float(librosa.midi_to_hz(fallback_midi))

        best_midi = fallback_midi
        best_score = float("-inf")
        for midi_candidate in candidate_pool:
            fundamental = float(librosa.midi_to_hz(midi_candidate))
            harmonic_score = 0.0
            for harmonic in range(1, 6):
                target_frequency = fundamental * harmonic
                if target_frequency > frequencies[-1]:
                    break
                nearest_bin = int(np.argmin(np.abs(frequencies - target_frequency)))
                local_start = max(0, nearest_bin - 1)
                local_end = min(len(spectrum), nearest_bin + 2)
                harmonic_score += float(np.max(spectrum[local_start:local_end])) / harmonic
            if harmonic_score > best_score:
                best_score = harmonic_score
                best_midi = midi_candidate

        return best_midi, float(librosa.midi_to_hz(best_midi))

    for start_index, end_index in zip(onset_list, onset_list[1:]):
        onset_segment_frames = frame_pitch_debug[start_index:end_index]
        if not onset_segment_frames:
            continue

        for segment_frames in _split_segment_by_pitch_continuity(onset_segment_frames):
            all_voiced_frames = [
                frame for frame in segment_frames
                if bool(frame["voiced"]) and frame["frequency_hz"] is not None
            ]
            if not all_voiced_frames:
                continue

            onset_supported = any(
                bool(frame.get("onset_peak")) and float(frame.get("onset_strength", 0.0)) >= 0.24
                for frame in all_voiced_frames[: min(4, len(all_voiced_frames))]
            )
            segment_span = float(all_voiced_frames[-1]["time"]) - float(all_voiced_frames[0]["time"])
            max_onset_strength = max(float(frame.get("onset_strength", 0.0)) for frame in all_voiced_frames)
            segment_midis = [int(frame["midi"]) for frame in all_voiced_frames if frame.get("midi") is not None]
            midi_histogram: dict[int, int] = {}
            for midi_note in segment_midis:
                midi_histogram[midi_note] = midi_histogram.get(midi_note, 0) + 1
            dominant_pitch_ratio = (
                max(midi_histogram.values()) / len(segment_midis)
                if segment_midis else 0.0
            )
            median_segment_confidence = float(
                np.median([float(frame["confidence"]) for frame in all_voiced_frames])
            )
            segment_has_delayed_onset = any(
                bool(frame.get("onset_peak")) and float(frame.get("onset_strength", 0.0)) >= 0.08
                for frame in all_voiced_frames
            )
            high_conf_voiced_frames = [
                frame for frame in all_voiced_frames
                if float(frame["confidence"]) >= 0.55
            ]
            relaxed_attack_frames = [
                frame for frame in all_voiced_frames[: min(6, len(all_voiced_frames))]
                if (
                    float(frame["confidence"]) >= 0.18
                    or (
                        bool(frame.get("onset_peak"))
                        and float(frame.get("onset_strength", 0.0)) >= 0.22
                    )
                )
            ]
            attack_onset_strength = max(
                (float(frame.get("onset_strength", 0.0)) for frame in all_voiced_frames[: min(4, len(all_voiced_frames))]),
                default=0.0,
            )
            allow_single_frame_onset_note = (
                onset_supported
                and len(relaxed_attack_frames) == 1
                and attack_onset_strength >= 0.32
            )
            coherent_short_pitch_island = (
                not high_conf_voiced_frames
                and len(all_voiced_frames) >= 8
                and 0.09 <= segment_span <= 0.24
                and dominant_pitch_ratio >= 0.8
                and median_segment_confidence >= 0.22
                and (
                    segment_has_delayed_onset
                    or (
                        len(all_voiced_frames) >= 12
                        and dominant_pitch_ratio >= 0.95
                        and median_segment_confidence >= 0.32
                    )
                )
            )

            if high_conf_voiced_frames:
                voiced_frames = list(high_conf_voiced_frames)
            elif onset_supported and (len(relaxed_attack_frames) >= 2 or allow_single_frame_onset_note):
                voiced_frames = relaxed_attack_frames
            elif coherent_short_pitch_island:
                voiced_frames = list(all_voiced_frames)
                onset_supported = True
            else:
                continue

            short_attack_note = (
                len(voiced_frames) <= 4
                or (onset_supported and len(all_voiced_frames) <= 6 and segment_span <= 0.14)
            )
            stable_frames = voiced_frames if short_attack_note else (voiced_frames[1:] if len(voiced_frames) >= 3 else voiced_frames)
            if not stable_frames:
                stable_frames = voiced_frames

            midi_candidates = np.array([int(frame["midi"]) for frame in stable_frames if frame["midi"] is not None], dtype=int)
            confidence_values = np.array([float(frame["confidence"]) for frame in stable_frames], dtype=float)
            if midi_candidates.size == 0:
                continue

            candidate_scores: dict[int, float] = {}
            for frame_index, (midi_note, confidence) in enumerate(zip(midi_candidates, confidence_values)):
                attack_weight = 1.0
                if short_attack_note and frame_index <= 1:
                    attack_weight = 1.35
                candidate_scores[midi_note] = candidate_scores.get(midi_note, 0.0) + float(confidence) * attack_weight
            frame_best_midi = max(candidate_scores.items(), key=lambda item: item[1])[0]
            if allowed_midis:
                frame_best_midi = min(
                    allowed_midis,
                    key=lambda midi_note: (abs(midi_note - frame_best_midi), abs(librosa.midi_to_hz(midi_note) - np.median([float(frame["frequency_hz"]) for frame in stable_frames]))),
                )

            start_frame = all_voiced_frames[0] if onset_supported else voiced_frames[0]
            start_time = float(start_frame["time"])
            offset_supported = any(
                bool(frame.get("offset_peak")) and float(frame.get("offset_strength", 0.0)) >= 0.12
                for frame in all_voiced_frames[max(0, len(all_voiced_frames) - 2):]
            )
            max_offset_strength = max(float(frame.get("offset_strength", 0.0)) for frame in all_voiced_frames)
            end_padding = 0.5 if offset_supported else 1.0
            end_time = float(all_voiced_frames[-1]["time"] + end_padding * hop_length / sample_rate)
            duration = max(0.05, end_time - start_time)
            harmonic_best_midi, harmonic_frequency = estimate_segment_pitch(start_time, end_time, frame_best_midi)
            best_midi = harmonic_best_midi
            frequency_hz = harmonic_frequency
            if abs(harmonic_best_midi - frame_best_midi) <= 2:
                best_midi = int(round((harmonic_best_midi + frame_best_midi) / 2))
                frequency_hz = float(librosa.midi_to_hz(best_midi))

            attack_frames = [
                frame for frame in all_voiced_frames[: min(4, len(all_voiced_frames))]
                if frame["midi"] is not None and float(frame["confidence"]) >= 0.78
            ]
            short_leading_note = duration <= 0.16 and len(voiced_frames) <= 4
            if short_leading_note and not attack_frames and onset_supported:
                attack_frames = [
                    frame for frame in all_voiced_frames[: min(4, len(all_voiced_frames))]
                    if frame["midi"] is not None and float(frame["confidence"]) >= 0.18
                ]
            if short_leading_note and attack_frames:
                attack_scores: dict[int, float] = {}
                for frame in attack_frames:
                    midi_note = int(frame["midi"])
                    confidence = float(frame["confidence"])
                    attack_scores[midi_note] = attack_scores.get(midi_note, 0.0) + confidence * 1.45
                attack_frame_best = max(attack_scores.items(), key=lambda item: item[1])[0]
                attack_best_midi, attack_frequency = estimate_segment_pitch(
                    start_time,
                    min(end_time, start_time + 0.09),
                    attack_frame_best,
                    attack_focus=True,
                )
                if allowed_midis:
                    attack_best_midi = min(
                        allowed_midis,
                        key=lambda midi_note: (
                            abs(midi_note - attack_best_midi),
                            abs(
                                librosa.midi_to_hz(midi_note)
                                - np.median([float(frame["frequency_hz"]) for frame in attack_frames if frame["frequency_hz"] is not None])
                            ),
                        ),
                    )
                    attack_frequency = float(librosa.midi_to_hz(attack_best_midi))

                attack_confidence = float(np.median([float(frame["confidence"]) for frame in attack_frames]))
                if (
                    attack_confidence >= 0.86
                    and attack_best_midi != best_midi
                    and abs(attack_best_midi - best_midi) <= 5
                ):
                    best_midi = attack_best_midi
                    frequency_hz = attack_frequency
            if allowed_midis:
                best_midi = min(
                    allowed_midis,
                    key=lambda midi_note: (abs(midi_note - best_midi), abs(librosa.midi_to_hz(midi_note) - frequency_hz)),
                )
                frequency_hz = float(librosa.midi_to_hz(best_midi))
            confidence = float(np.nanmedian(confidence_values)) if confidence_values.size else 0.0
            observed_source_frames = attack_frames if short_leading_note and attack_frames else all_voiced_frames
            observed_frequency_hz = float(
                np.nanmedian(
                    [float(frame["frequency_hz"]) for frame in observed_source_frames if frame["frequency_hz"] is not None]
                )
            ) if observed_source_frames else frequency_hz

            ultra_short_attack_note = (
                duration <= 0.08
                and onset_supported
                and max_onset_strength >= 0.4
                and observed_frequency_hz > 0
            )
            if ultra_short_attack_note:
                observed_midi = float(librosa.hz_to_midi(observed_frequency_hz))
                if abs(observed_midi - best_midi) >= 3.5:
                    best_midi = min(
                        candidate_pool,
                        key=lambda midi_note: abs(midi_note - observed_midi),
                    )
                    frequency_hz = float(librosa.midi_to_hz(best_midi))

            short_note_without_onset = (
                duration <= 0.14
                and len(voiced_frames) <= 3
                and not onset_supported
                and confidence < 0.92
            )
            if short_note_without_onset:
                continue

            notes.append(
                MelodyNote(
                    note_index=len(notes) + 1,
                    start_time=round(start_time, 4),
                    end_time=round(end_time, 4),
                    duration=round(duration, 4),
                    frequency_hz=round(frequency_hz, 3),
                    midi=int(best_midi),
                    note_name=_format_note_name(int(best_midi)),
                    confidence=round(confidence, 4),
                    observed_frequency_hz=round(observed_frequency_hz, 3),
                    raw_beat_position=0.0,
                    quantized_beat_position=0.0,
                    quantized_duration_beats=0.0,
                    measure_number=0,
                    beat_in_measure=0.0,
                    onset_supported=onset_supported,
                    offset_supported=offset_supported,
                    onset_strength=round(max_onset_strength, 4),
                    offset_strength=round(max_offset_strength, 4),
                )
            )

    for index, note in enumerate(notes, start=1):
        note.note_index = index
    return notes


def _copy_note(note: MelodyNote) -> MelodyNote:
    return MelodyNote(**note.to_dict())


def _notes_close_in_pitch(left: MelodyNote, right: MelodyNote, semitone_threshold: float = 0.5) -> bool:
    return abs(left.midi - right.midi) <= semitone_threshold


def _observed_midi_estimate(note: MelodyNote) -> float:
    observed_frequency = note.observed_frequency_hz or note.frequency_hz
    if observed_frequency and observed_frequency > 0:
        return float(librosa.hz_to_midi(observed_frequency))
    return float(note.midi)


def _pitch_candidates_for_note(
    note: MelodyNote,
    allowed_midis: list[int] | None,
) -> list[int]:
    candidate_pool = allowed_midis if allowed_midis else UKULELE_PLAYABLE_MIDI
    observed_midi = _observed_midi_estimate(note)
    if (
        note.confidence >= 0.9
        and note.duration >= 0.14
        and abs(observed_midi - note.midi) <= 0.8
    ):
        return [int(note.midi)]

    search_radius = 4.5
    existing_radius = 3.5
    if note.confidence >= 0.82:
        search_radius = 2.5
        existing_radius = 2.0
    if note.onset_supported and note.duration <= 0.16:
        search_radius = max(search_radius, 3.0)
        existing_radius = max(existing_radius, 3.0)

    nearby = [
        midi_note
        for midi_note in candidate_pool
        if abs(midi_note - observed_midi) <= search_radius or abs(midi_note - note.midi) <= existing_radius
    ]
    if len(nearby) < 4:
        nearest = sorted(candidate_pool, key=lambda midi_note: abs(midi_note - observed_midi))[:6]
        nearby.extend(nearest)
    nearby.append(note.midi)
    return sorted(set(int(midi_note) for midi_note in nearby))


def _optimize_pitch_path(
    notes: list[MelodyNote],
    allowed_midis: list[int] | None,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    optimized = [_copy_note(note) for note in notes]
    candidate_lists = [_pitch_candidates_for_note(note, allowed_midis) for note in optimized]
    dp_scores: list[dict[int, float]] = []
    parents: list[dict[int, int | None]] = []

    for index, note in enumerate(optimized):
        observed_midi = _observed_midi_estimate(note)
        duration_weight = 1.25 if note.duration <= 0.18 else 1.0
        if note.onset_supported and note.duration <= 0.16:
            duration_weight += 0.2

        current_scores: dict[int, float] = {}
        current_parents: dict[int, int | None] = {}
        for midi_candidate in candidate_lists[index]:
            observation_penalty = abs(observed_midi - midi_candidate) * duration_weight
            existing_penalty = abs(note.midi - midi_candidate) * (0.18 if note.confidence >= 0.88 else 0.08)
            onset_bonus = -0.12 if note.onset_supported and abs(observed_midi - midi_candidate) <= 1.0 else 0.0
            base_score = observation_penalty + existing_penalty + onset_bonus

            if index == 0:
                current_scores[midi_candidate] = base_score
                current_parents[midi_candidate] = None
                continue

            best_score = float("inf")
            best_parent: int | None = None
            previous_note = optimized[index - 1]
            for previous_candidate, previous_score in dp_scores[index - 1].items():
                interval = abs(midi_candidate - previous_candidate)
                transition_penalty = interval * 0.05 + max(0, interval - 5) * 0.09
                if midi_candidate == previous_candidate and note.onset_supported and note.duration <= 0.14:
                    transition_penalty += 0.06
                if midi_candidate == previous_candidate and not note.onset_supported:
                    transition_penalty -= 0.05
                if previous_note.onset_supported and note.onset_supported and 1 <= interval <= 2:
                    transition_penalty -= 0.04
                if note.duration >= 0.24 and interval >= 8:
                    transition_penalty += 0.12
                score = previous_score + base_score + transition_penalty
                if score < best_score:
                    best_score = score
                    best_parent = previous_candidate

            current_scores[midi_candidate] = best_score
            current_parents[midi_candidate] = best_parent

        dp_scores.append(current_scores)
        parents.append(current_parents)

    final_candidate = min(dp_scores[-1], key=dp_scores[-1].get)
    chosen_path = [final_candidate]
    for index in range(len(optimized) - 1, 0, -1):
        parent_candidate = parents[index][chosen_path[-1]]
        if parent_candidate is None:
            parent_candidate = optimized[index - 1].midi
        chosen_path.append(parent_candidate)
    chosen_path.reverse()

    for note, chosen_midi in zip(optimized, chosen_path):
        if (
            note.confidence >= 0.92
            and note.duration >= 0.14
            and abs(_observed_midi_estimate(note) - note.midi) <= 0.8
        ):
            continue
        note.midi = int(chosen_midi)
        note.note_name = _format_note_name(note.midi)
        note.frequency_hz = round(float(librosa.midi_to_hz(note.midi)), 3)

    return optimized


def _build_window_signature(
    notes: list[MelodyNote],
    window_start_beat: float,
) -> list[tuple[float, float, int]]:
    ordered = sorted(notes, key=lambda item: item.quantized_beat_position)
    if not ordered:
        return []

    observed_midis = [int(round(_observed_midi_estimate(note))) for note in ordered]
    anchor_pitch = int(round(np.median(observed_midis)))
    signature: list[tuple[float, float, int]] = []
    for note in ordered:
        relative_position = round(note.quantized_beat_position - window_start_beat, 4)
        duration = round(note.quantized_duration_beats, 4)
        pitch_interval = int(round(_observed_midi_estimate(note))) - anchor_pitch
        signature.append((relative_position, duration, pitch_interval))
    return signature


def _window_signature_similarity(
    left_signature: list[tuple[float, float, int]],
    right_signature: list[tuple[float, float, int]],
) -> float:
    if not left_signature or not right_signature:
        return float("-inf")

    count_gap = abs(len(left_signature) - len(right_signature))
    if count_gap > max(2, int(max(len(left_signature), len(right_signature)) * 0.5)):
        return float("-inf")

    left_length = len(left_signature)
    right_length = len(right_signature)
    gap_penalty = 0.42
    dp = [[float("inf")] * (right_length + 1) for _ in range(left_length + 1)]
    dp[0][0] = 0.0
    for left_index in range(1, left_length + 1):
        dp[left_index][0] = left_index * gap_penalty
    for right_index in range(1, right_length + 1):
        dp[0][right_index] = right_index * gap_penalty

    for left_index in range(1, left_length + 1):
        for right_index in range(1, right_length + 1):
            left_position, left_duration, left_interval = left_signature[left_index - 1]
            right_position, right_duration, right_interval = right_signature[right_index - 1]
            match_cost = (
                abs(left_position - right_position) * 1.45
                + abs(left_duration - right_duration) * 0.9
                + min(abs(left_interval - right_interval), 10) * 0.06
            )
            dp[left_index][right_index] = min(
                dp[left_index - 1][right_index - 1] + match_cost,
                dp[left_index - 1][right_index] + gap_penalty,
                dp[left_index][right_index - 1] + gap_penalty,
            )

    average_cost = dp[left_length][right_length] / max(left_length, right_length)
    return 1.35 - average_cost


def _repeated_phrase_consistency_score(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> float:
    if len(notes) < 8:
        return 0.0

    last_measure = max(note.measure_number for note in notes)
    similarity_scores: list[float] = []
    for window_size in (2, 1):
        windows: list[tuple[int, list[tuple[float, float, int]]]] = []
        for start_measure in range(1, last_measure - window_size + 2):
            end_measure = start_measure + window_size - 1
            window_notes = [
                note
                for note in notes
                if start_measure <= note.measure_number <= end_measure
            ]
            if len(window_notes) < max(3, window_size * 2):
                continue
            window_start_beat = (start_measure - 1) * beats_per_measure
            signature = _build_window_signature(window_notes, window_start_beat)
            windows.append((start_measure, signature))

        for left_index in range(len(windows)):
            for right_index in range(left_index + 1, len(windows)):
                left_measure, left_signature = windows[left_index]
                right_measure, right_signature = windows[right_index]
                if right_measure - left_measure < window_size:
                    continue
                similarity = _window_signature_similarity(left_signature, right_signature)
                if similarity > 0.2:
                    similarity_scores.append(similarity + (0.08 if window_size == 2 else 0.0))

    if not similarity_scores:
        return 0.0

    top_scores = sorted(similarity_scores, reverse=True)[:6]
    return float(np.mean(top_scores))


def _build_hypothesis_notes(
    notes: list[MelodyNote],
    quantized_positions: list[float],
    beat_duration: float,
    beats_per_measure: int,
) -> list[MelodyNote]:
    hypothesis_notes = [_copy_note(note) for note in notes]
    for note, snapped_position in zip(hypothesis_notes, quantized_positions):
        note.quantized_beat_position = round(snapped_position, 4)
        note.quantized_duration_beats = round(_quantize_duration_to_grid(note.duration / beat_duration), 4)
        note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
        note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)
    return hypothesis_notes


def _regularize_repeated_phrase_windows(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 8:
        return notes

    regularized = [_copy_note(note) for note in notes]
    last_measure = max(note.measure_number for note in regularized)

    def mode_count(values: list[int]) -> int:
        count_map: dict[int, int] = {}
        for value in values:
            count_map[value] = count_map.get(value, 0) + 1
        return max(count_map.items(), key=lambda item: (item[1], -item[0]))[0]

    def estimate_window_beat_seconds(window_notes: list[MelodyNote], window_start_beat: float) -> float:
        seconds_per_beat_candidates: list[float] = []
        for note in window_notes:
            if note.quantized_duration_beats >= 0.25:
                seconds_per_beat_candidates.append(note.duration / max(note.quantized_duration_beats, 0.25))
        ordered_notes = sorted(window_notes, key=lambda item: item.start_time)
        for previous_note, current_note in zip(ordered_notes, ordered_notes[1:]):
            beat_gap = current_note.quantized_beat_position - previous_note.quantized_beat_position
            time_gap = current_note.start_time - previous_note.start_time
            if beat_gap >= 0.24 and time_gap > 0.03:
                seconds_per_beat_candidates.append(time_gap / beat_gap)
        if not seconds_per_beat_candidates:
            return 0.3
        return float(max(0.18, min(0.9, np.median(seconds_per_beat_candidates))))

    def estimate_window_start_time(
        window_notes: list[MelodyNote],
        window_start_beat: float,
        seconds_per_beat: float,
    ) -> float:
        estimates = [
            note.start_time - (note.quantized_beat_position - window_start_beat) * seconds_per_beat
            for note in window_notes
        ]
        return float(np.median(estimates)) if estimates else 0.0

    def align_notes_to_consensus(
        window_notes: list[MelodyNote],
        window_start_beat: float,
        consensus_relative_positions: list[float],
    ) -> tuple[list[MelodyNote | None], list[MelodyNote]]:
        ordered_notes = sorted(window_notes, key=lambda item: item.quantized_beat_position)
        matches: list[MelodyNote | None] = [None] * len(consensus_relative_positions)
        used_indices: set[int] = set()
        next_slot = 0
        for note in ordered_notes:
            relative_position = note.quantized_beat_position - window_start_beat
            candidate_slots = [
                slot_index
                for slot_index in range(next_slot, len(consensus_relative_positions))
                if slot_index not in used_indices
            ]
            if not candidate_slots:
                continue
            best_slot = min(
                candidate_slots,
                key=lambda slot_index: abs(consensus_relative_positions[slot_index] - relative_position),
            )
            if abs(consensus_relative_positions[best_slot] - relative_position) <= 0.34:
                matches[best_slot] = note
                used_indices.add(best_slot)
                next_slot = best_slot + 1
        unmatched = [
            note for note in ordered_notes
            if all(note is not match for match in matches if match is not None)
        ]
        return matches, unmatched

    for window_size in (2, 1):
        windows: list[dict[str, object]] = []
        for start_measure in range(1, last_measure - window_size + 2):
            end_measure = start_measure + window_size - 1
            window_notes = [
                note
                for note in regularized
                if start_measure <= note.measure_number <= end_measure
            ]
            if len(window_notes) < max(3, window_size * 2):
                continue
            window_start_beat = (start_measure - 1) * beats_per_measure
            signature = _build_window_signature(window_notes, window_start_beat)
            windows.append(
                {
                    "start_measure": start_measure,
                    "window_start_beat": window_start_beat,
                    "notes": sorted(window_notes, key=lambda item: item.quantized_beat_position),
                    "signature": signature,
                }
            )

        used_windows: set[int] = set()
        for left_index in range(len(windows)):
            if left_index in used_windows:
                continue
            left_window = windows[left_index]
            group = [left_window]
            similarity_threshold = 0.74 if window_size == 2 else 0.68
            for right_index in range(left_index + 1, len(windows)):
                right_window = windows[right_index]
                similarity = _window_signature_similarity(
                    left_window["signature"],
                    right_window["signature"],
                )
                if (
                    similarity >= similarity_threshold
                    and abs(len(left_window["notes"]) - len(right_window["notes"])) <= 2
                ):
                    group.append(right_window)
                    used_windows.add(right_index)

            if len(group) < 2:
                continue
            pair_mode = len(group) == 2
            if pair_mode:
                pair_similarity = _window_signature_similarity(
                    group[0]["signature"],
                    group[1]["signature"],
                )
                if pair_similarity < 0.84:
                    continue

            note_counts = [len(window["notes"]) for window in group]
            if len(group) >= 3:
                reference_count = mode_count(note_counts)
            else:
                reference_count = max(note_counts)
            reference_windows = [window for window in group if len(window["notes"]) == reference_count]
            if len(reference_windows) < 2 and len(group) >= 2:
                reference_windows = sorted(group, key=lambda window: len(window["notes"]), reverse=True)[:2]
                reference_count = max(len(window["notes"]) for window in reference_windows)
            if len(reference_windows) < 2:
                continue

            reference_spread_ok = True
            for note_index in range(reference_count):
                relative_positions = [
                    window["notes"][note_index].quantized_beat_position - window["window_start_beat"]
                    for window in reference_windows
                ]
                if max(relative_positions) - min(relative_positions) > 0.45:
                    reference_spread_ok = False
                    break
            if not reference_spread_ok:
                continue

            consensus_relative_positions: list[float] = []
            consensus_durations: list[float] = []
            consensus_midis: list[int] = []
            consensus_templates: list[MelodyNote] = []
            consensus_support_counts: list[int] = []
            for note_index in range(reference_count):
                supporting_notes = [window["notes"][note_index] for window in reference_windows]
                relative_positions = [
                    note.quantized_beat_position - reference_windows[0]["window_start_beat"]
                    for note in supporting_notes
                ]
                durations = [note.quantized_duration_beats for note in supporting_notes]
                supporting_midis = [
                    int(round(_observed_midi_estimate(note)))
                    if note.observed_frequency_hz > 0 else note.midi
                    for note in supporting_notes
                ]
                consensus_relative_positions.append(round(float(np.median(relative_positions)) * 4) / 4)
                consensus_durations.append(_quantize_duration_to_grid(float(np.median(durations))))
                consensus_midis.append(int(round(np.median(supporting_midis))))
                consensus_templates.append(max(supporting_notes, key=lambda note: (note.confidence, note.duration)))
                consensus_support_counts.append(len(supporting_notes))

            for window in group:
                window_notes: list[MelodyNote] = sorted(window["notes"], key=lambda item: item.quantized_beat_position)
                window_start_beat = float(window["window_start_beat"])
                matches, unmatched_notes = align_notes_to_consensus(
                    window_notes,
                    window_start_beat,
                    consensus_relative_positions,
                )
                missing_count = reference_count - len(window_notes)
                can_insert_short_consensus = (
                    missing_count == 1
                    if pair_mode else
                    1 <= missing_count <= 2
                )
                seconds_per_beat = estimate_window_beat_seconds(window_notes, window_start_beat)
                window_start_time = estimate_window_start_time(window_notes, window_start_beat, seconds_per_beat)

                for note_index in range(reference_count):
                    consensus_relative_position = consensus_relative_positions[note_index]
                    consensus_duration = consensus_durations[note_index]
                    consensus_midi = consensus_midis[note_index]
                    template_note = consensus_templates[note_index]
                    matched_note = matches[note_index]

                    if matched_note is not None:
                        position_delta = abs(
                            (matched_note.quantized_beat_position - window_start_beat)
                            - consensus_relative_position
                        )
                        duration_delta = abs(matched_note.quantized_duration_beats - consensus_duration)
                        should_reposition_matched = not pair_mode and not (
                            matched_note.confidence >= 0.9
                            and position_delta <= 0.20
                            and duration_delta <= 0.25
                        )
                        if should_reposition_matched:
                            matched_note.quantized_beat_position = round(window_start_beat + consensus_relative_position, 4)
                            matched_note.quantized_duration_beats = round(consensus_duration, 4)
                            matched_note.measure_number = max(
                                1,
                                int(np.floor(matched_note.quantized_beat_position / beats_per_measure)) + 1,
                            )
                            matched_note.beat_in_measure = round(
                                (matched_note.quantized_beat_position % beats_per_measure) + 1,
                                4,
                            )

                        observed_midi = _observed_midi_estimate(matched_note)
                        pitch_delta = abs(matched_note.midi - consensus_midi)
                        observed_delta = abs(observed_midi - consensus_midi)
                        if (
                            pitch_delta >= 2
                            and (
                                matched_note.confidence <= (0.82 if pair_mode else 0.88)
                                or observed_delta <= 1.2
                            )
                        ):
                            matched_note.midi = int(consensus_midi)
                            matched_note.note_name = _format_note_name(matched_note.midi)
                            matched_note.frequency_hz = round(float(librosa.midi_to_hz(matched_note.midi)), 3)
                        continue

                    if not (
                        can_insert_short_consensus
                        and consensus_support_counts[note_index] >= 2
                        and consensus_duration <= (0.5 if pair_mode else 0.75)
                    ):
                        continue

                    nearby_unmatched = [
                        note for note in unmatched_notes
                        if abs((note.quantized_beat_position - window_start_beat) - consensus_relative_position) <= 0.38
                    ]
                    if nearby_unmatched:
                        recovered = min(
                            nearby_unmatched,
                            key=lambda note: (
                                abs((note.quantized_beat_position - window_start_beat) - consensus_relative_position),
                                abs(_observed_midi_estimate(note) - consensus_midi),
                            ),
                        )
                        recovered.quantized_beat_position = round(window_start_beat + consensus_relative_position, 4)
                        recovered.quantized_duration_beats = round(consensus_duration, 4)
                        recovered.measure_number = max(
                            1,
                            int(np.floor(recovered.quantized_beat_position / beats_per_measure)) + 1,
                        )
                        recovered.beat_in_measure = round(
                            (recovered.quantized_beat_position % beats_per_measure) + 1,
                            4,
                        )
                        recovered.midi = int(consensus_midi)
                        recovered.note_name = _format_note_name(recovered.midi)
                        recovered.frequency_hz = round(float(librosa.midi_to_hz(recovered.midi)), 3)
                        unmatched_notes = [note for note in unmatched_notes if note is not recovered]
                        continue

                    inserted = _copy_note(template_note)
                    inserted.note_index = 0
                    inserted.quantized_beat_position = round(window_start_beat + consensus_relative_position, 4)
                    inserted.quantized_duration_beats = round(consensus_duration, 4)
                    inserted.measure_number = max(
                        1,
                        int(np.floor(inserted.quantized_beat_position / beats_per_measure)) + 1,
                    )
                    inserted.beat_in_measure = round(
                        (inserted.quantized_beat_position % beats_per_measure) + 1,
                        4,
                    )
                    inserted.midi = int(consensus_midi)
                    inserted.note_name = _format_note_name(inserted.midi)
                    inserted.frequency_hz = round(float(librosa.midi_to_hz(inserted.midi)), 3)
                    inserted.observed_frequency_hz = inserted.frequency_hz
                    inserted.confidence = round(min(template_note.confidence, 0.74), 4)
                    inserted.onset_supported = True
                    inserted.offset_supported = False
                    inserted.onset_strength = round(max(template_note.onset_strength, 0.22), 4)
                    inserted.offset_strength = 0.0
                    inserted.start_time = round(
                        max(0.0, window_start_time + consensus_relative_position * seconds_per_beat),
                        4,
                    )
                    inserted.duration = round(max(0.05, consensus_duration * seconds_per_beat), 4)
                    inserted.end_time = round(inserted.start_time + inserted.duration, 4)
                    regularized.append(inserted)

    regularized = sorted(
        regularized,
        key=lambda note: (note.quantized_beat_position, note.measure_number, note.beat_in_measure, note.start_time),
    )
    for index, note in enumerate(regularized, start=1):
        note.note_index = index

    return regularized


def _merge_and_filter_notes(
    raw_notes: list[MelodyNote],
    min_duration_seconds: float = 0.12,
    merge_gap_seconds: float = 0.09,
) -> tuple[list[MelodyNote], dict[str, int | float | bool]]:
    if not raw_notes:
        return [], {
            "raw_note_count": 0,
            "merged_fragments": 0,
            "filtered_short_notes": 0,
            "final_note_count": 0,
            "average_note_duration": 0.0,
            "dense_repeat_warning": False,
        }

    merged: list[MelodyNote] = [_copy_note(raw_notes[0])]
    merged_fragments = 0
    filtered_short_notes = 0

    for raw_index, current in enumerate(raw_notes[1:], start=1):
        previous = merged[-1]
        gap = round(current.start_time - previous.end_time, 4)
        same_or_close_pitch = _notes_close_in_pitch(previous, current)
        next_raw = raw_notes[raw_index + 1] if raw_index + 1 < len(raw_notes) else None
        previous_raw = raw_notes[raw_index - 1] if raw_index - 1 >= 0 else None
        repeated_run_context = (
            previous.midi == current.midi
            and (
                (next_raw is not None and next_raw.midi == current.midi)
                or (previous_raw is not None and previous_raw.midi == current.midi)
            )
        )
        plausible_short_reattack = (
            previous.midi == current.midi
            and previous.duration >= 0.1
            and current.duration >= 0.1
            and previous.confidence >= 0.88
            and current.confidence >= 0.88
            and gap <= 0.03
        )
        onset_offset_supported_reattack = (
            previous.midi == current.midi
            and gap <= 0.04
            and current.onset_supported
            and current.onset_strength >= 0.24
            and (
                previous.offset_supported
                or previous.offset_strength >= 0.1
                or current.offset_strength >= 0.08
            )
        )
        likely_reattack_same_note = (
            previous.midi == current.midi
            and previous.duration >= 0.16
            and current.duration >= 0.16
            and gap <= merge_gap_seconds
        )
        if (
            same_or_close_pitch
            and gap <= merge_gap_seconds
            and not likely_reattack_same_note
            and not plausible_short_reattack
            and not onset_offset_supported_reattack
            and not repeated_run_context
        ):
            previous.end_time = round(current.end_time, 4)
            previous.duration = round(previous.end_time - previous.start_time, 4)
            previous.frequency_hz = round((previous.frequency_hz + current.frequency_hz) / 2, 3)
            previous.midi = int(round((previous.midi + current.midi) / 2))
            previous.note_name = _format_note_name(previous.midi)
            previous.confidence = round(max(previous.confidence, current.confidence), 4)
            merged_fragments += 1
            continue
        merged.append(_copy_note(current))

    stabilized: list[MelodyNote] = []
    for index, note in enumerate(merged):
        previous_merged = merged[index - 1] if index > 0 else None
        next_merged = merged[index + 1] if index + 1 < len(merged) else None
        previous_anchor = previous_merged
        previous_anchor_index = index - 1
        while previous_anchor is not None and previous_anchor.duration < 0.08 and previous_anchor_index > 0:
            previous_anchor_index -= 1
            previous_anchor = merged[previous_anchor_index]
        repeated_same_pitch_context = (
            (previous_merged is not None and previous_merged.midi == note.midi)
            or (next_merged is not None and next_merged.midi == note.midi)
        )
        neighboring_gap_is_tight = (
            (previous_merged is not None and note.start_time - previous_merged.end_time <= 0.05)
            or (next_merged is not None and next_merged.start_time - note.end_time <= 0.05)
        )

        preserve_short_attack_note = (
            note.duration >= 0.08
            and note.confidence >= 0.9
            and (
                (previous_merged is not None and previous_merged.midi != note.midi)
                or (next_merged is not None and next_merged.midi != note.midi)
            )
        )
        preserve_repeated_short_note = (
            note.duration >= 0.08
            and repeated_same_pitch_context
            and neighboring_gap_is_tight
            and note.confidence >= 0.88
            and (note.onset_supported or note.onset_strength >= 0.24)
        )
        preserve_leading_pickup_note = (
            0.07 <= note.duration <= 0.16
            and note.confidence >= 0.9
            and next_merged is not None
            and note.midi != next_merged.midi
            and (next_merged.start_time - note.end_time) <= 0.04
            and next_merged.duration >= max(0.24, note.duration * 1.6)
            and next_merged.confidence >= 0.88
            and not repeated_same_pitch_context
            and (
                previous_merged is None
                or previous_merged.midi != note.midi
                or note.start_time - previous_merged.end_time > 0.04
            )
            and note.onset_supported
        )
        preserve_trailing_short_note = (
            0.08 <= note.duration <= 0.16
            and previous_anchor is not None
            and next_merged is not None
            and previous_anchor.midi != note.midi
            and note.start_time - previous_merged.end_time <= 0.06
            and next_merged.start_time - note.end_time <= 0.12
            and previous_anchor.duration >= max(0.16, note.duration * 1.5)
            and previous_anchor.confidence >= 0.84
            and not repeated_same_pitch_context
            and (
                note.confidence >= 0.78
                or (
                    note.onset_supported
                    and note.onset_strength >= 0.45
                    and note.observed_frequency_hz > 0
                )
            )
            and (
                note.onset_supported
                or note.onset_strength >= 0.22
                or abs(note.frequency_hz - previous_anchor.frequency_hz) >= 30.0
            )
        )
        preserve_embedded_contrast_tail = (
            0.08 <= note.duration <= 0.18
            and previous_merged is not None
            and next_merged is not None
            and previous_merged.midi == next_merged.midi
            and note.midi != previous_merged.midi
            and note.start_time - previous_merged.end_time <= 0.14
            and next_merged.start_time - note.end_time <= 0.12
            and previous_merged.duration >= 0.16
            and next_merged.duration >= 0.16
            and previous_merged.confidence >= 0.84
            and next_merged.confidence >= 0.84
            and note.observed_frequency_hz > 0
            and abs(librosa.hz_to_midi(note.observed_frequency_hz) - previous_merged.midi) >= 1.4
            and (
                note.confidence >= 0.65
                or (
                    note.onset_supported
                    and note.onset_strength >= 0.24
                )
                or abs(note.frequency_hz - previous_merged.frequency_hz) >= 28.0
            )
        )
        preserve_bridge_short_note = (
            0.07 <= note.duration <= 0.16
            and previous_anchor is not None
            and next_merged is not None
            and note.midi != previous_anchor.midi
            and note.midi != next_merged.midi
            and previous_anchor.duration >= 0.14
            and next_merged.duration >= 0.28
            and note.start_time - previous_anchor.end_time <= 0.16
            and next_merged.start_time - note.end_time <= 0.14
            and note.observed_frequency_hz > 0
            and abs(librosa.hz_to_midi(note.observed_frequency_hz) - note.midi) <= 1.2
            and abs(note.midi - previous_anchor.midi) >= 1
            and abs(note.midi - next_merged.midi) >= 1
            and (
                note.confidence >= 0.62
                or note.onset_strength >= 0.42
            )
        )
        preserve_stepwise_short_connector = (
            0.05 <= note.duration <= 0.16
            and previous_merged is not None
            and next_merged is not None
            and note.midi != previous_merged.midi
            and note.midi != next_merged.midi
            and note.start_time - previous_merged.end_time <= 0.12
            and next_merged.start_time - note.end_time <= 0.14
            and note.observed_frequency_hz > 0
            and abs(librosa.hz_to_midi(note.observed_frequency_hz) - note.midi) <= 1.6
            and (
                note.onset_supported
                or note.onset_strength >= 0.22
                or note.confidence >= 0.72
            )
            and (
                (
                    (note.midi - previous_merged.midi) * (next_merged.midi - note.midi) > 0
                    and abs(note.midi - previous_merged.midi) <= 4
                    and abs(next_merged.midi - note.midi) <= 4
                )
                or (
                    previous_merged.midi == next_merged.midi
                    and abs(note.midi - previous_merged.midi) >= 2
                )
            )
        )
        preserve_same_pitch_dotted_tail = (
            0.04 <= note.duration <= 0.12
            and previous_anchor is not None
            and previous_anchor.midi == note.midi
            and previous_anchor.duration >= 0.18
            and previous_anchor.confidence >= 0.84
            and previous_merged is not None
            and previous_merged.midi == note.midi
            and note.start_time - previous_merged.end_time <= 0.05
            and next_merged is not None
            and next_merged.midi != note.midi
            and next_merged.start_time - note.end_time <= 0.14
            and (
                note.onset_supported
                or note.onset_strength >= 0.34
                or note.confidence >= 0.92
            )
            and (
                note.observed_frequency_hz <= 0
                or abs(librosa.hz_to_midi(note.observed_frequency_hz) - note.midi) <= 1.1
            )
        )
        preserve_terminal_contrast_note = (
            0.06 <= note.duration <= 0.18
            and previous_anchor is not None
            and note.midi != previous_anchor.midi
            and previous_anchor.duration >= 0.14
            and note.start_time - previous_anchor.end_time <= 0.14
            and (
                next_merged is None
                or next_merged.start_time - note.end_time >= 0.16
            )
            and note.observed_frequency_hz > 0
            and (
                note.confidence >= 0.58
                or note.onset_supported
                or note.onset_strength >= 0.3
            )
            and (
                abs(librosa.hz_to_midi(note.observed_frequency_hz) - note.midi) <= 1.4
                or abs(note.frequency_hz - previous_anchor.frequency_hz) >= 25.0
            )
        )
        if preserve_trailing_short_note and note.observed_frequency_hz > 0:
            observed_midi = float(librosa.hz_to_midi(note.observed_frequency_hz))
            if abs(observed_midi - note.midi) >= 3.5:
                corrected_midi = min(
                    UKULELE_PLAYABLE_MIDI,
                    key=lambda midi_note: abs(midi_note - observed_midi),
                )
                note.midi = int(corrected_midi)
                note.note_name = _format_note_name(note.midi)
                note.frequency_hz = round(float(librosa.midi_to_hz(note.midi)), 3)
        if preserve_stepwise_short_connector and note.observed_frequency_hz > 0:
            observed_midi = float(librosa.hz_to_midi(note.observed_frequency_hz))
            if abs(observed_midi - note.midi) >= 1.8:
                local_candidates = [
                    midi_note
                    for midi_note in UKULELE_PLAYABLE_MIDI
                    if abs(midi_note - observed_midi) <= 3.5
                    or abs(midi_note - previous_merged.midi) <= 4
                    or abs(midi_note - next_merged.midi) <= 4
                ]
                if local_candidates:
                    corrected_midi = min(local_candidates, key=lambda midi_note: abs(midi_note - observed_midi))
                    note.midi = int(corrected_midi)
                    note.note_name = _format_note_name(note.midi)
                    note.frequency_hz = round(float(librosa.midi_to_hz(note.midi)), 3)
        if (
            note.duration >= min_duration_seconds
            or preserve_short_attack_note
            or preserve_repeated_short_note
            or preserve_leading_pickup_note
            or preserve_trailing_short_note
            or preserve_embedded_contrast_tail
            or preserve_bridge_short_note
            or preserve_stepwise_short_connector
            or preserve_same_pitch_dotted_tail
            or preserve_terminal_contrast_note
        ):
            stabilized.append(note)
            continue

        attached = False
        if (
            not preserve_leading_pickup_note
            and not preserve_trailing_short_note
            and not preserve_embedded_contrast_tail
            and not preserve_bridge_short_note
            and not preserve_stepwise_short_connector
            and not preserve_same_pitch_dotted_tail
            and not preserve_terminal_contrast_note
            and stabilized
            and _notes_close_in_pitch(stabilized[-1], note, semitone_threshold=1)
        ):
            stabilized[-1].end_time = round(note.end_time, 4)
            stabilized[-1].duration = round(stabilized[-1].end_time - stabilized[-1].start_time, 4)
            stabilized[-1].confidence = round(max(stabilized[-1].confidence, note.confidence), 4)
            filtered_short_notes += 1
            attached = True
        elif (
            not preserve_leading_pickup_note
            and not preserve_trailing_short_note
            and not preserve_embedded_contrast_tail
            and not preserve_bridge_short_note
            and not preserve_stepwise_short_connector
            and not preserve_same_pitch_dotted_tail
            and not preserve_terminal_contrast_note
            and next_merged is not None
            and _notes_close_in_pitch(note, next_merged, semitone_threshold=1)
        ):
            next_merged.start_time = round(note.start_time, 4)
            next_merged.duration = round(next_merged.end_time - next_merged.start_time, 4)
            next_merged.confidence = round(max(next_merged.confidence, note.confidence), 4)
            filtered_short_notes += 1
            attached = True
        if attached:
            continue
        filtered_short_notes += 1

    for index, note in enumerate(stabilized, start=1):
        note.note_index = index
    average_duration = round(
        float(np.mean([note.duration for note in stabilized])) if stabilized else 0.0,
        4,
    )
    dense_repeat_warning = False
    repeated = 0
    for previous, current in zip(stabilized, stabilized[1:]):
        if previous.midi == current.midi and current.start_time - previous.end_time <= 0.12:
            repeated += 1
    if repeated >= 2:
        dense_repeat_warning = True

    return stabilized, {
        "raw_note_count": len(raw_notes),
        "merged_fragments": merged_fragments,
        "filtered_short_notes": filtered_short_notes,
        "final_note_count": len(stabilized),
        "average_note_duration": average_duration,
        "dense_repeat_warning": dense_repeat_warning,
    }


def _recover_high_register_bridge_notes(
    notes: list[MelodyNote],
    frame_pitch_debug: list[dict[str, object]],
    sample_rate: int,
    allowed_midis: list[int] | None = None,
) -> list[MelodyNote]:
    if len(notes) < 2 or not frame_pitch_debug:
        return notes

    # Recover short bridge notes from frame-level pitch islands that sit between
    # two stronger neighboring anchors. Although this started from high-register
    # failures, the same pattern appears in mid-register 3-5-7 / 7-5-3 figures.
    candidate_pool = list(allowed_midis or UKULELE_PLAYABLE_MIDI)
    if not candidate_pool:
        return notes

    hop_seconds = HOP_LENGTH / sample_rate
    recovered = [_copy_note(note) for note in notes]

    def _frame_runs(frames: list[dict[str, object]]) -> list[list[dict[str, object]]]:
        runs: list[list[dict[str, object]]] = []
        current_run: list[dict[str, object]] = []
        for frame in frames:
            if not bool(frame.get("voiced")) or frame.get("midi") is None:
                if current_run:
                    runs.append(current_run)
                    current_run = []
                continue

            if not current_run:
                current_run = [frame]
                continue

            previous = current_run[-1]
            close_pitch = abs(int(frame["midi"]) - int(previous["midi"])) <= 1
            close_time = float(frame["time"]) - float(previous["time"]) <= hop_seconds * 2.5
            if close_pitch and close_time:
                current_run.append(frame)
            else:
                runs.append(current_run)
                current_run = [frame]

        if current_run:
            runs.append(current_run)
        return runs

    inserted: list[MelodyNote] = []
    for index, left_note in enumerate(recovered[:-1]):
        right_note = recovered[index + 1]

        gap = right_note.start_time - left_note.end_time
        if gap < 0.08 or gap > 0.32:
            continue

        search_start = left_note.end_time + hop_seconds * 0.25
        search_end = right_note.start_time - hop_seconds * 0.25
        candidate_frames = [
            frame
            for frame in frame_pitch_debug
            if search_start <= float(frame["time"]) <= search_end
        ]
        if not candidate_frames:
            continue

        best_note: MelodyNote | None = None
        best_score = float("-inf")
        for run in _frame_runs(candidate_frames):
            run_start = float(run[0]["time"])
            run_end = float(run[-1]["time"]) + hop_seconds
            run_duration = max(0.05, run_end - run_start)
            if run_duration < 0.05 or run_duration > 0.18:
                continue

            median_confidence = float(np.median([float(frame["confidence"]) for frame in run]))
            max_onset_strength = max(float(frame.get("onset_strength", 0.0)) for frame in run)
            onset_supported = any(
                bool(frame.get("onset_frame")) or bool(frame.get("onset_peak"))
                for frame in run
            )
            if not onset_supported or max_onset_strength < 0.18:
                continue

            observed_frequencies = [
                float(frame["frequency_hz"])
                for frame in run
                if frame.get("frequency_hz") is not None
            ]
            if not observed_frequencies:
                continue

            observed_frequency_hz = float(np.median(observed_frequencies))
            observed_midi = float(librosa.hz_to_midi(observed_frequency_hz))
            candidate_midi = min(candidate_pool, key=lambda midi_note: abs(midi_note - observed_midi))
            lower_neighbor = min(left_note.midi, right_note.midi)
            upper_neighbor = max(left_note.midi, right_note.midi)
            same_pitch_anchor_island = (
                left_note.midi == right_note.midi
                and candidate_midi >= left_note.midi + 2
                and run_duration <= 0.14
            )
            if not same_pitch_anchor_island and not (lower_neighbor + 1 <= candidate_midi <= upper_neighbor - 1):
                continue

            dominant_midis = [int(frame["midi"]) for frame in run if frame.get("midi") is not None]
            stability_ratio = 0.0
            if dominant_midis:
                midi_histogram: dict[int, int] = {}
                for midi_note in dominant_midis:
                    midi_histogram[midi_note] = midi_histogram.get(midi_note, 0) + 1
                stability_ratio = max(midi_histogram.values()) / len(dominant_midis)

            if median_confidence < 0.12 and max_onset_strength < 0.3:
                continue
            if stability_ratio < 0.7:
                continue

            score = (
                median_confidence
                + max_onset_strength
                + stability_ratio * 0.4
                - abs(candidate_midi - observed_midi) * 0.08
            )
            if score <= best_score:
                continue

            best_score = score
            best_note = MelodyNote(
                note_index=0,
                start_time=round(run_start, 4),
                end_time=round(run_end, 4),
                duration=round(run_duration, 4),
                frequency_hz=round(float(librosa.midi_to_hz(candidate_midi)), 3),
                midi=int(candidate_midi),
                note_name=_format_note_name(int(candidate_midi)),
                confidence=round(median_confidence, 4),
                observed_frequency_hz=round(observed_frequency_hz, 3),
                raw_beat_position=0.0,
                quantized_beat_position=0.0,
                quantized_duration_beats=0.0,
                measure_number=0,
                beat_in_measure=0.0,
                onset_supported=onset_supported,
                offset_supported=False,
                onset_strength=round(max_onset_strength, 4),
                offset_strength=0.0,
            )

        if best_note is not None:
            inserted.append(best_note)

    if not inserted:
        return recovered

    combined = sorted([*recovered, *inserted], key=lambda item: (item.start_time, item.end_time))
    for note_index, note in enumerate(combined, start=1):
        note.note_index = note_index
    return combined


def _recover_same_pitch_reattacks(
    notes: list[MelodyNote],
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    recovered: list[MelodyNote] = []
    index = 0
    while index < len(notes):
        current = _copy_note(notes[index])
        run = [current]
        next_index = index + 1

        while next_index < len(notes):
            candidate = notes[next_index]
            gap = candidate.start_time - run[-1].end_time
            if candidate.midi != current.midi or gap > 0.09:
                break
            run.append(_copy_note(candidate))
            next_index += 1

        if len(run) == 1:
            recovered.append(current)
            index = next_index
            continue

        onset_count = sum(1 for note in run if note.onset_supported or note.onset_strength >= 0.28)
        if onset_count < 2:
            recovered.extend(run)
            index = next_index
            continue

        clusters: list[list[MelodyNote]] = [[run[0]]]
        for note in run[1:]:
            starts_new_cluster = (
                (note.onset_supported or note.onset_strength >= 0.28)
                and note.start_time - clusters[-1][-1].end_time <= 0.09
            )
            if starts_new_cluster:
                clusters.append([note])
            else:
                clusters[-1].append(note)

        if len(clusters) == 1:
            recovered.extend(run)
            index = next_index
            continue

        for cluster in clusters:
            cluster_start = cluster[0].start_time
            cluster_end = cluster[-1].end_time
            cluster_duration = round(max(0.05, cluster_end - cluster_start), 4)
            observed_values = [
                note.observed_frequency_hz
                for note in cluster
                if note.observed_frequency_hz and note.observed_frequency_hz > 0
            ]
            recovered.append(
                MelodyNote(
                    note_index=0,
                    start_time=round(cluster_start, 4),
                    end_time=round(cluster_end, 4),
                    duration=cluster_duration,
                    frequency_hz=round(float(librosa.midi_to_hz(current.midi)), 3),
                    midi=current.midi,
                    note_name=_format_note_name(current.midi),
                    confidence=round(max(note.confidence for note in cluster), 4),
                    observed_frequency_hz=round(float(np.median(observed_values)), 3) if observed_values else current.observed_frequency_hz,
                    raw_beat_position=0.0,
                    quantized_beat_position=0.0,
                    quantized_duration_beats=0.0,
                    measure_number=0,
                    beat_in_measure=0.0,
                    onset_supported=bool(cluster[0].onset_supported),
                    offset_supported=bool(cluster[-1].offset_supported),
                    onset_strength=round(max(note.onset_strength for note in cluster), 4),
                    offset_strength=round(max(note.offset_strength for note in cluster), 4),
                )
            )

        index = next_index

    for note_index, note in enumerate(recovered, start=1):
        note.note_index = note_index
    return recovered




def _consolidate_same_pitch_fragments(
    notes: list[MelodyNote],
    bpm_estimate: float,
    fragment_gap_seconds: float = 0.03,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
    target_unit_seconds = beat_duration / 4
    if target_unit_seconds <= 0:
        return notes

    consolidated: list[MelodyNote] = []
    index = 0
    while index < len(notes):
        current = _copy_note(notes[index])
        next_index = index + 1

        while next_index < len(notes):
            candidate = notes[next_index]
            gap = candidate.start_time - current.end_time
            if candidate.midi != current.midi or gap > fragment_gap_seconds:
                break

            previous_same = index > 0 and notes[index - 1].midi == current.midi
            next_same = next_index + 1 < len(notes) and notes[next_index + 1].midi == current.midi
            repeated_run_context = previous_same or next_same
            if repeated_run_context:
                break
            onset_offset_boundary = (
                candidate.onset_supported
                and candidate.onset_strength >= 0.24
                and (current.offset_supported or current.offset_strength >= 0.1)
            )
            if onset_offset_boundary:
                break

            combined_duration = candidate.end_time - current.start_time
            current_is_fragment = current.duration <= target_unit_seconds * 0.8
            candidate_is_fragment = candidate.duration <= target_unit_seconds * 0.8
            combined_near_unit = combined_duration <= target_unit_seconds * 1.35
            confidence_ok = current.confidence >= 0.88 and candidate.confidence >= 0.88
            no_new_attack_continuation = (
                gap <= 0.01
                and not candidate.onset_supported
                and candidate.onset_strength < 0.6
                and candidate.duration <= target_unit_seconds * 0.7
                and combined_duration <= beat_duration * 0.7
                and current.duration >= target_unit_seconds * 1.1
                and current.onset_supported
                and current.confidence >= 0.84
                and candidate.confidence >= 0.72
            )
            explicit_same_pitch_reattack = (
                gap <= 0.09
                and candidate.onset_supported
                and candidate.onset_strength >= 0.24
                and current.duration >= max(0.12, target_unit_seconds * 0.75)
                and candidate.duration >= 0.05
            )

            if (
                not explicit_same_pitch_reattack
                and (
                    (confidence_ok and combined_near_unit and (current_is_fragment or candidate_is_fragment))
                    or no_new_attack_continuation
                )
            ):
                current.end_time = round(candidate.end_time, 4)
                current.duration = round(current.end_time - current.start_time, 4)
                current.confidence = round(max(current.confidence, candidate.confidence), 4)
                next_index += 1
                continue
            break

        consolidated.append(current)
        index = next_index

    for note_index, note in enumerate(consolidated, start=1):
        note.note_index = note_index
    return consolidated


def _notes_for_rhythm_inference(notes: list[MelodyNote]) -> list[MelodyNote]:
    if len(notes) < 4:
        return [_copy_note(note) for note in notes]

    filtered: list[MelodyNote] = []
    for index, note in enumerate(notes):
        previous_note = notes[index - 1] if index - 1 >= 0 else None
        next_note = notes[index + 1] if index + 1 < len(notes) else None
        looks_like_pickup = (
            0.07 <= note.duration <= 0.16
            and note.confidence >= 0.9
            and next_note is not None
            and note.midi != next_note.midi
            and (next_note.start_time - note.end_time) <= 0.04
            and next_note.duration >= max(0.24, note.duration * 1.6)
            and next_note.confidence >= 0.88
        )
        looks_like_bridge_island = (
            previous_note is not None
            and next_note is not None
            and 0.09 <= note.duration <= 0.22
            and note.confidence <= 0.75
            and note.midi != previous_note.midi
            and note.midi != next_note.midi
            and (note.start_time - previous_note.end_time) <= 0.12
            and (next_note.start_time - note.end_time) <= 0.12
            and previous_note.confidence >= note.confidence
            and next_note.confidence >= note.confidence
        )
        if looks_like_pickup or looks_like_bridge_island:
            continue
        filtered.append(_copy_note(note))

    return filtered if len(filtered) >= 4 else [_copy_note(note) for note in notes]


def _contextual_short_note_pitch_correction(
    notes: list[MelodyNote],
    allowed_midis: list[int] | None,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    corrected = [_copy_note(note) for note in notes]
    candidate_universe = allowed_midis or UKULELE_PLAYABLE_MIDI

    def score_candidate(
        midi_candidate: int,
        note: MelodyNote,
        previous_note: MelodyNote,
        next_note: MelodyNote,
    ) -> float:
        observed_frequency = note.observed_frequency_hz or note.frequency_hz or float(librosa.midi_to_hz(note.midi))
        observed_distance = abs(librosa.hz_to_midi(observed_frequency) - midi_candidate)
        transition_cost = (abs(midi_candidate - previous_note.midi) + abs(next_note.midi - midi_candidate)) * 0.18
        repeated_anchor_bonus = 0.0
        if previous_note.midi == next_note.midi and midi_candidate != previous_note.midi:
            repeated_anchor_bonus -= 0.45
        if midi_candidate == previous_note.midi == next_note.midi:
            repeated_anchor_bonus += 0.7
        if midi_candidate == previous_note.midi != next_note.midi:
            repeated_anchor_bonus += 0.18
        if midi_candidate == next_note.midi != previous_note.midi:
            repeated_anchor_bonus += 0.12
        return observed_distance * 1.2 + transition_cost + repeated_anchor_bonus

    for index in range(1, len(corrected) - 1):
        note = corrected[index]
        previous_note = corrected[index - 1]
        next_note = corrected[index + 1]

        short_context_note = (
            note.duration <= 0.18
            and note.confidence >= 0.72
            and previous_note.duration >= max(0.18, note.duration * 1.25)
            and next_note.duration >= max(0.18, note.duration * 1.25)
            and (
                note.onset_supported
                or note.onset_strength >= 0.24
                or (
                    note.observed_frequency_hz > 0
                    and abs(librosa.hz_to_midi(note.observed_frequency_hz) - note.midi) <= 1.6
                )
            )
        )
        if not short_context_note:
            continue

        candidate_pool = [
            midi_candidate
            for midi_candidate in candidate_universe
            if abs(midi_candidate - note.midi) <= 6
            or abs(midi_candidate - previous_note.midi) <= 5
            or abs(midi_candidate - next_note.midi) <= 5
        ]
        if not candidate_pool:
            candidate_pool = candidate_universe

        current_score = score_candidate(note.midi, note, previous_note, next_note)
        best_candidate = note.midi
        best_score = current_score
        for midi_candidate in candidate_pool:
            candidate_score = score_candidate(midi_candidate, note, previous_note, next_note)
            if candidate_score < best_score:
                best_score = candidate_score
                best_candidate = midi_candidate

        if best_candidate != note.midi and best_score <= current_score - 0.55:
            note.midi = int(best_candidate)
            note.note_name = _format_note_name(note.midi)
            note.frequency_hz = round(float(librosa.midi_to_hz(note.midi)), 3)

    return corrected


def _quantize_notes(
    notes: list[MelodyNote],
    bpm_estimate: float,
    beats_per_measure: int,
    grid_origin_seconds: float,
) -> list[MelodyNote]:
    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
    quantized: list[MelodyNote] = []
    for raw_note in notes:
        note = _copy_note(raw_note)
        raw_beat_position = (note.start_time - grid_origin_seconds) / beat_duration
        quantized_beat_position = _snap_to_sixteenth_grid(raw_beat_position)
        quantized_duration_beats = _quantize_duration_to_grid(note.duration / beat_duration)
        measure_number = max(1, int(np.floor(quantized_beat_position / beats_per_measure)) + 1)
        beat_in_measure = (quantized_beat_position % beats_per_measure) + 1
        note.raw_beat_position = round(raw_beat_position, 4)
        note.quantized_beat_position = round(quantized_beat_position, 4)
        note.quantized_duration_beats = round(quantized_duration_beats, 4)
        note.measure_number = measure_number
        note.beat_in_measure = round(beat_in_measure, 4)
        note.rhythm_value = _rhythm_value_from_duration(note.quantized_duration_beats)
        note.beam_group = None
        quantized.append(note)
    return quantized


def _apply_triplet_aware_quantization(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if not notes:
        return notes

    adjusted = [_copy_note(note) for note in notes]
    raw_beat_groups: dict[int, list[MelodyNote]] = {}
    for note in adjusted:
        beat_bucket = int(np.floor(note.raw_beat_position + 1e-6))
        raw_beat_groups.setdefault(beat_bucket, []).append(note)

    for beat_bucket, beat_notes in raw_beat_groups.items():
        beat_notes = sorted(beat_notes, key=lambda item: item.raw_beat_position)
        offsets = [note.raw_beat_position - beat_bucket for note in beat_notes]

        if len(beat_notes) == 3 and _supports_triplet_interpretation(beat_notes, offsets):
            triplet_positions = [beat_bucket, beat_bucket + 1 / 3, beat_bucket + 2 / 3]
            for note, snapped_position in zip(beat_notes, triplet_positions):
                note.quantized_beat_position = round(snapped_position, 4)
                note.quantized_duration_beats = round(1 / 3, 4)
                note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
                note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)
            continue

        if len(beat_notes) == 6:
            first_triplet = beat_notes[:3]
            second_triplet = beat_notes[3:]
            first_offsets = [note.raw_beat_position - beat_bucket for note in first_triplet]
            second_offsets = [note.raw_beat_position - beat_bucket for note in second_triplet]
            if (
                _supports_triplet_interpretation(first_triplet, [0.0, 1 / 3, 2 / 3])
                or (_match_offsets(first_offsets, [0.0, 1 / 6, 1 / 3], tolerance=0.14) and len({note.midi for note in first_triplet}) == 1)
            ) and (
                _supports_triplet_interpretation(second_triplet, [offset - 0.5 for offset in second_offsets])
                or (_match_offsets(second_offsets, [0.5, 2 / 3, 5 / 6], tolerance=0.14) and len({note.midi for note in second_triplet}) == 1)
            ):
                snapped_positions = [
                    beat_bucket + 0.0,
                    beat_bucket + 1 / 3,
                    beat_bucket + 2 / 3,
                    beat_bucket + 1.0,
                    beat_bucket + 1 + 1 / 3,
                    beat_bucket + 1 + 2 / 3,
                ]
                for note, snapped_position in zip(beat_notes, snapped_positions):
                    note.quantized_beat_position = round(snapped_position, 4)
                    note.quantized_duration_beats = round(1 / 3, 4)
                    note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
                    note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)

    # Phrase-level fallback: repeated same-pitch runs often represent one or two triplet groups
    # even when frame segmentation and beat phase are slightly off.
    run_start = 0
    while run_start < len(adjusted):
        run_end = run_start + 1
        while run_end < len(adjusted) and adjusted[run_end].midi == adjusted[run_start].midi:
            run_end += 1

        run = adjusted[run_start:run_end]
        run_length = len(run)
        run_span_beats = run[-1].raw_beat_position - run[0].raw_beat_position if run_length >= 2 else 0.0
        anchor_beat = round(run[0].raw_beat_position)

        if run_length == 3 and 0.6 <= run_span_beats <= 1.1:
            snapped_positions = [anchor_beat, anchor_beat + 1 / 3, anchor_beat + 2 / 3]
            for note, snapped_position in zip(run, snapped_positions):
                note.quantized_beat_position = round(snapped_position, 4)
                note.quantized_duration_beats = round(1 / 3, 4)
                note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
                note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)

        if run_length == 6 and 1.4 <= run_span_beats <= 2.2:
            snapped_positions = [
                anchor_beat,
                anchor_beat + 1 / 3,
                anchor_beat + 2 / 3,
                anchor_beat + 1.0,
                anchor_beat + 1 + 1 / 3,
                anchor_beat + 1 + 2 / 3,
            ]
            for note, snapped_position in zip(run, snapped_positions):
                note.quantized_beat_position = round(snapped_position, 4)
                note.quantized_duration_beats = round(1 / 3, 4)
                note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
                note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)

        run_start = run_end

    return adjusted


def _set_note_quantization(
    note: MelodyNote,
    snapped_position: float,
    duration_beats: float,
    beats_per_measure: int,
) -> None:
    note.quantized_beat_position = round(snapped_position, 4)
    note.quantized_duration_beats = round(duration_beats, 4)
    note.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
    note.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)


def _best_non_triplet_template(
    beat_notes: list[MelodyNote],
    beat_bucket: int,
    allow_dotted: bool = False,
) -> tuple[str, list[float], list[float]] | None:
    if len(beat_notes) < 2 or len(beat_notes) > 4:
        return None

    raw_offsets = [max(0.0, min(0.99, note.raw_beat_position - beat_bucket)) for note in beat_notes]
    raw_durations = [max(0.05, note.duration) for note in beat_notes]
    current_positions = [note.quantized_beat_position for note in beat_notes]
    has_collision = len({round(value, 4) for value in current_positions}) < len(current_positions)

    templates: list[tuple[str, list[float], list[float]]] = []
    if len(beat_notes) == 2:
        d1, d2 = raw_durations
        raw_gap = raw_offsets[1] - raw_offsets[0]
        late_tail_evidence = (
            raw_offsets[0] <= 0.12
            and
            0.62 <= raw_gap <= 0.92
            and d1 >= max(d2 * 1.85, 0.18)
            and d2 <= 0.18
            and (
                beat_notes[1].onset_supported
                or beat_notes[1].onset_strength >= 0.26
                or beat_notes[1].confidence >= 0.88
            )
        )
        early_tail_evidence = (
            raw_offsets[0] <= 0.12
            and
            0.18 <= raw_gap <= 0.32
            and d2 >= max(d1 * 1.85, 0.18)
            and d1 <= 0.18
            and (
                beat_notes[0].onset_supported
                or beat_notes[0].onset_strength >= 0.26
                or beat_notes[0].confidence >= 0.88
            )
        )

        templates = [("two_eighths", [0.0, 0.5], [0.5, 0.5])]
        if allow_dotted and late_tail_evidence:
            templates.append(("dotted_eighth_then_sixteenth", [0.0, 0.75], [0.75, 0.25]))
        if allow_dotted and early_tail_evidence:
            templates.append(("sixteenth_then_dotted_eighth", [0.0, 0.25], [0.25, 0.75]))
    elif len(beat_notes) == 3:
        templates = [
            ("front_sixteenth_then_eighth", [0.0, 0.25, 0.5], [0.25, 0.25, 0.5]),
            ("front_eighth_then_sixteenth", [0.0, 0.5, 0.75], [0.5, 0.25, 0.25]),
        ]
    elif len(beat_notes) == 4:
        templates = [("four_sixteenths", [0.0, 0.25, 0.5, 0.75], [0.25, 0.25, 0.25, 0.25])]

    if not templates:
        return None

    def duration_penalty(pattern_name: str) -> float:
        penalty = 0.0
        if len(raw_durations) == 2:
            d1, d2 = raw_durations
            if pattern_name == "dotted_eighth_then_sixteenth":
                if d1 <= d2 * 1.55:
                    penalty += 0.28
                if (raw_offsets[1] - raw_offsets[0]) <= 0.58:
                    penalty += 0.16
            if pattern_name == "sixteenth_then_dotted_eighth":
                if d2 <= d1 * 1.55:
                    penalty += 0.28
                if (raw_offsets[1] - raw_offsets[0]) >= 0.42:
                    penalty += 0.16
            return penalty
        if len(raw_durations) != 3:
            return 0.0
        d1, d2, d3 = raw_durations
        if pattern_name == "front_sixteenth_then_eighth":
            if d3 <= max(d1, d2) * 1.08:
                penalty += 0.2
            if d1 >= d3 * 0.95 or d2 >= d3 * 0.95:
                penalty += 0.1
        if pattern_name == "front_eighth_then_sixteenth":
            if d1 <= max(d2, d3) * 1.08:
                penalty += 0.2
            if d2 >= d1 * 0.95 or d3 >= d1 * 0.95:
                penalty += 0.1
        return penalty

    best: tuple[str, list[float], list[float]] | None = None
    best_score = float("inf")
    for pattern_name, template_offsets, template_durations in templates:
        position_score = sum((raw_offset - template_offset) ** 2 for raw_offset, template_offset in zip(raw_offsets, template_offsets))
        collision_bonus = -0.08 if has_collision else 0.0
        score = position_score + duration_penalty(pattern_name) + collision_bonus
        if score < best_score:
            best_score = score
            best = (pattern_name, template_offsets, template_durations)

    return best


def _resolve_non_triplet_subbeat_patterns(
    notes: list[MelodyNote],
    beats_per_measure: int,
    allow_dotted: bool = False,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    resolved = [_copy_note(note) for note in notes]
    raw_beat_groups: dict[int, list[MelodyNote]] = {}
    for note in resolved:
        beat_bucket = int(np.floor(note.raw_beat_position + 1e-6))
        raw_beat_groups.setdefault(beat_bucket, []).append(note)

    for beat_bucket, beat_notes in raw_beat_groups.items():
        beat_notes = sorted(beat_notes, key=lambda item: item.raw_beat_position)
        if any(abs(note.quantized_duration_beats - (1 / 3)) <= 0.03 for note in beat_notes):
            continue

        template = _best_non_triplet_template(beat_notes, beat_bucket, allow_dotted=allow_dotted)
        if template is None:
            continue

        pattern_name, template_offsets, template_durations = template
        current_positions = [note.quantized_beat_position for note in beat_notes]
        expected_positions = [beat_bucket + offset for offset in template_offsets]
        should_adjust = (
            len({round(value, 4) for value in current_positions}) < len(current_positions)
            or any(abs(current - expected) > 0.17 for current, expected in zip(current_positions, expected_positions))
        )
        if not should_adjust:
            continue

        for note, snapped_position, duration_beats in zip(beat_notes, expected_positions, template_durations):
            _set_note_quantization(note, snapped_position, duration_beats, beats_per_measure)

    return resolved


def _repair_boundary_reactivation_artifacts(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 4:
        return notes

    repaired: list[MelodyNote] = []
    index = 0
    copied = [_copy_note(note) for note in notes]

    while index < len(copied):
        current_note = copied[index]
        beat_bucket = int(np.floor(current_note.raw_beat_position + 1e-6))
        group_end = index + 1
        while (
            group_end < len(copied)
            and int(np.floor(copied[group_end].raw_beat_position + 1e-6)) == beat_bucket
        ):
            group_end += 1

        beat_notes = copied[index:group_end]
        if len(beat_notes) == 4 and not any(abs(note.quantized_duration_beats - (1 / 3)) <= 0.03 for note in beat_notes):
            first, second, third, fourth = beat_notes
            raw_offsets = [note.raw_beat_position - beat_bucket for note in beat_notes]
            prefix_template = _best_non_triplet_template(beat_notes[:3], beat_bucket, allow_dotted=False)

            looks_like_boundary_reactivation = (
                prefix_template is not None
                and prefix_template[0] in {"front_sixteenth_then_eighth", "front_eighth_then_sixteenth"}
                and raw_offsets[3] >= 0.92
                and fourth.duration >= max(0.6, second.duration * 1.6, third.duration * 2.0)
                and second.midi == fourth.midi
                and third.midi != fourth.midi
                and third.duration <= 0.18
                and third.confidence >= 0.65
                and fourth.onset_strength <= 0.45
            )

            if looks_like_boundary_reactivation:
                _, prefix_offsets, prefix_durations = prefix_template
                for note, snapped_offset, duration_beats in zip(
                    beat_notes[:3],
                    prefix_offsets,
                    prefix_durations,
                ):
                    _set_note_quantization(note, beat_bucket + snapped_offset, duration_beats, beats_per_measure)
                fourth_target = beat_bucket + 1.0
                fourth.quantized_beat_position = round(fourth_target, 4)
                fourth.quantized_duration_beats = max(1.0, fourth.quantized_duration_beats)
                fourth.measure_number = max(1, int(np.floor(fourth_target / beats_per_measure)) + 1)
                fourth.beat_in_measure = round((fourth_target % beats_per_measure) + 1, 4)
                repaired.extend(beat_notes)
                index = group_end
                continue

        repaired.extend(beat_notes)
        index = group_end

    return repaired


def _resolve_monophonic_mixed_subdivision_collisions(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    corrected = [_copy_note(note) for note in notes]
    index = 0
    while index + 2 < len(corrected):
        window = corrected[index : index + 3]
        if len({note.measure_number for note in window}) != 1:
            index += 1
            continue
        if any(abs(note.quantized_duration_beats - (1 / 3)) <= 0.03 for note in window):
            index += 1
            continue

        raw_positions = [note.raw_beat_position for note in window]
        raw_span = raw_positions[-1] - raw_positions[0]
        if raw_span > 1.08:
            index += 1
            continue

        quantized_positions = [round(note.quantized_beat_position, 4) for note in window]
        quantized_unique = sorted(set(quantized_positions))
        if len(quantized_unique) != 2:
            index += 1
            continue

        if any((right - left) <= 0 for left, right in zip(raw_positions, raw_positions[1:])):
            index += 1
            continue

        raw_durations = [max(note.duration, 0.05) for note in window]
        snapped_first = _snap_to_sixteenth_grid(raw_positions[0])
        base_candidates = sorted(
            {
                float(np.floor(raw_positions[0])),
                float(np.ceil(raw_positions[0])),
                float(round(raw_positions[0])),
                float(np.floor(np.mean(raw_positions))),
                float(np.ceil(np.mean(raw_positions))),
                float(snapped_first),
                float(snapped_first - 0.25),
                float(snapped_first + 0.25),
            }
        )
        templates = [
            ("front_sixteenth_then_eighth", [0.0, 0.25, 0.5], [0.25, 0.25, 0.5]),
            ("front_eighth_then_sixteenth", [0.0, 0.5, 0.75], [0.5, 0.25, 0.25]),
        ]

        best_template: tuple[str, list[float], list[float], float] | None = None
        for pattern_name, offsets, durations in templates:
            for base in base_candidates:
                absolute_positions = [base + offset for offset in offsets]
                fit_error = float(
                    np.mean([abs(actual - target) for actual, target in zip(raw_positions, absolute_positions)])
                )
                duration_penalty = 0.0
                if pattern_name == "front_sixteenth_then_eighth":
                    if raw_durations[2] <= max(raw_durations[0], raw_durations[1]) * 1.05:
                        duration_penalty += 0.18
                    if raw_durations[0] >= raw_durations[2] * 1.15:
                        duration_penalty += 0.08
                else:
                    if raw_durations[0] <= max(raw_durations[1], raw_durations[2]) * 1.05:
                        duration_penalty += 0.18
                    if raw_durations[2] >= raw_durations[0] * 1.15:
                        duration_penalty += 0.08

                collision_penalty = 0.0
                if pattern_name == "front_sixteenth_then_eighth" and abs(quantized_positions[0] - quantized_positions[1]) > 0.01:
                    collision_penalty += 0.04
                if pattern_name == "front_eighth_then_sixteenth" and abs(quantized_positions[1] - quantized_positions[2]) > 0.01:
                    collision_penalty += 0.04

                total_score = fit_error + duration_penalty + collision_penalty
                if best_template is None or total_score < best_template[3]:
                    best_template = (pattern_name, absolute_positions, durations, total_score)

        if best_template is None or best_template[3] > 0.36:
            index += 1
            continue

        _, snapped_positions, snapped_durations, _ = best_template
        for note, snapped_position, duration_beats in zip(window, snapped_positions, snapped_durations):
            _set_note_quantization(note, snapped_position, duration_beats, beats_per_measure)

        index += 3

    return corrected


def _repair_split_dotted_subdivision_groups(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    ordered = sorted([_copy_note(note) for note in notes], key=lambda item: item.quantized_beat_position)
    repaired: list[MelodyNote] = []
    index = 0

    while index < len(ordered):
        if index + 2 < len(ordered):
            first, second, third = ordered[index:index + 3]
            same_measure = len({first.measure_number, second.measure_number, third.measure_number}) == 1
            same_bucket = len({
                int(np.floor(first.quantized_beat_position + 1e-6)),
                int(np.floor(second.quantized_beat_position + 1e-6)),
                int(np.floor(third.quantized_beat_position + 1e-6)),
            }) == 1
            if same_measure and same_bucket:
                beat_bucket = int(np.floor(first.quantized_beat_position + 1e-6))
                first_offsets = [
                    round(first.quantized_beat_position - beat_bucket, 4),
                    round(second.quantized_beat_position - beat_bucket, 4),
                    round(third.quantized_beat_position - beat_bucket, 4),
                ]

                leading_split_dotted = (
                    first.midi == second.midi
                    and third.midi != first.midi
                    and abs(first_offsets[0] - 0.0) <= 0.08
                    and abs(first_offsets[1] - 0.5) <= 0.12
                    and abs(first_offsets[2] - 0.75) <= 0.12
                    and first.quantized_duration_beats >= 0.45
                    and second.quantized_duration_beats <= 0.26
                    and third.quantized_duration_beats <= 0.26
                    and (
                        not second.onset_supported
                        or second.onset_strength <= max(first.onset_strength + 0.12, 0.42)
                    )
                )
                trailing_split_dotted = (
                    first.midi != second.midi
                    and second.midi == third.midi
                    and abs(first_offsets[0] - 0.0) <= 0.08
                    and abs(first_offsets[1] - 0.25) <= 0.12
                    and abs(first_offsets[2] - 0.5) <= 0.12
                    and first.quantized_duration_beats <= 0.26
                    and second.quantized_duration_beats <= 0.26
                    and third.quantized_duration_beats >= 0.45
                    and (
                        not third.onset_supported
                        or third.onset_strength <= max(second.onset_strength + 0.12, 0.42)
                    )
                )

                if leading_split_dotted:
                    merged = _copy_note(first)
                    merged.end_time = round(second.end_time, 4)
                    merged.duration = round(merged.end_time - merged.start_time, 4)
                    merged.confidence = round(max(first.confidence, second.confidence), 4)
                    _set_note_quantization(merged, beat_bucket, 0.75, beats_per_measure)
                    repaired.append(merged)
                    short_tail = _copy_note(third)
                    _set_note_quantization(short_tail, beat_bucket + 0.75, 0.25, beats_per_measure)
                    repaired.append(short_tail)
                    index += 3
                    continue

                if trailing_split_dotted:
                    short_head = _copy_note(first)
                    _set_note_quantization(short_head, beat_bucket, 0.25, beats_per_measure)
                    repaired.append(short_head)
                    merged = _copy_note(second)
                    merged.start_time = round(second.start_time, 4)
                    merged.end_time = round(third.end_time, 4)
                    merged.duration = round(merged.end_time - merged.start_time, 4)
                    merged.confidence = round(max(second.confidence, third.confidence), 4)
                    _set_note_quantization(merged, beat_bucket + 0.25, 0.75, beats_per_measure)
                    repaired.append(merged)
                    index += 3
                    continue

        repaired.append(_copy_note(ordered[index]))
        index += 1

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _repair_same_pitch_dotted_tail_sequences(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    repaired = [_copy_note(note) for note in notes]
    for index in range(len(repaired) - 2):
        first = repaired[index]
        second = repaired[index + 1]
        third = repaired[index + 2]

        if first.midi != second.midi or third.midi == first.midi:
            continue

        same_or_adjacent_measure = abs(third.measure_number - first.measure_number) <= 1
        if not same_or_adjacent_measure:
            continue

        if first.quantized_duration_beats < 0.45 or second.duration > 0.12:
            continue

        if second.start_time - first.end_time > 0.05 or third.start_time - second.end_time > 0.14:
            continue

        if not (
            second.onset_supported
            or second.onset_strength >= 0.34
            or second.confidence >= 0.92
        ):
            continue

        if second.observed_frequency_hz > 0:
            observed_midi = float(librosa.hz_to_midi(second.observed_frequency_hz))
            if abs(observed_midi - second.midi) > 1.1:
                continue

        first_position = round(first.quantized_beat_position, 4)
        second_target = round(first_position + 0.75, 4)
        third_target = round(first_position + 1.0, 4)

        if abs(second.raw_beat_position - second_target) > 0.42:
            continue
        if abs(third.raw_beat_position - third_target) > 0.38:
            continue

        _set_note_quantization(first, first_position, 0.75, beats_per_measure)
        _set_note_quantization(second, second_target, 0.25, beats_per_measure)
        if abs(third.quantized_beat_position - third_target) <= 0.34:
            _set_note_quantization(third, third_target, max(third.quantized_duration_beats, 0.25), beats_per_measure)

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _repair_post_dotted_rearticulation(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    repaired = [_copy_note(note) for note in notes]
    dotted_patterns = {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}

    for index in range(len(repaired) - 2):
        first = repaired[index]
        second = repaired[index + 1]
        third = repaired[index + 2]

        if first.beat_pattern not in dotted_patterns or second.beat_pattern != first.beat_pattern:
            continue
        if first.beam_group is None or first.beam_group != second.beam_group:
            continue
        if third.midi != second.midi:
            continue

        raw_gap = third.raw_beat_position - second.raw_beat_position
        if not 0.26 <= raw_gap <= 0.52:
            continue
        if not (
            third.onset_supported
            or third.onset_strength >= 0.24
            or third.confidence >= 0.78
        ):
            continue

        next_beat_boundary = float(np.floor(second.raw_beat_position + 1e-6) + 1.0)
        if abs(third.raw_beat_position - next_beat_boundary) > 0.18:
            continue

        if third.quantized_beat_position >= second.quantized_beat_position + 0.18:
            continue

        snapped_position = next_beat_boundary
        snapped_duration = 0.5 if third.duration >= 0.18 else 0.25
        _set_note_quantization(third, snapped_position, snapped_duration, beats_per_measure)

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _recover_missing_notes_from_raw_clusters(
    quantized_notes: list[MelodyNote],
    raw_notes: list[MelodyNote],
    bpm_estimate: float,
    beats_per_measure: int,
    grid_origin_seconds: float,
) -> list[MelodyNote]:
    if len(raw_notes) < 2 or not quantized_notes:
        return quantized_notes

    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
    recovered = [_copy_note(note) for note in quantized_notes]

    def aggregate_cluster(cluster: list[MelodyNote]) -> MelodyNote:
        template = _copy_note(cluster[0])
        template.start_time = round(cluster[0].start_time, 4)
        template.end_time = round(cluster[-1].end_time, 4)
        template.duration = round(max(0.05, template.end_time - template.start_time), 4)
        template.confidence = round(max(note.confidence for note in cluster), 4)
        template.onset_supported = bool(cluster[0].onset_supported)
        template.offset_supported = bool(cluster[-1].offset_supported)
        template.onset_strength = round(max(note.onset_strength for note in cluster), 4)
        template.offset_strength = round(max(note.offset_strength for note in cluster), 4)
        observed_values = [
            note.observed_frequency_hz
            for note in cluster
            if note.observed_frequency_hz and note.observed_frequency_hz > 0
        ]
        if observed_values:
            template.observed_frequency_hz = round(float(np.median(observed_values)), 3)
        return template

    clusters: list[MelodyNote] = []
    current_cluster = [_copy_note(raw_notes[0])]
    for candidate in raw_notes[1:]:
        previous = current_cluster[-1]
        gap = candidate.start_time - previous.end_time
        starts_new_same_pitch_reattack = (
            candidate.midi == previous.midi
            and gap <= 0.09
            and (candidate.onset_supported or candidate.onset_strength >= 0.24)
        )
        continues_cluster = candidate.midi == previous.midi and gap <= 0.09 and not starts_new_same_pitch_reattack
        if continues_cluster:
            current_cluster.append(_copy_note(candidate))
            continue
        clusters.append(aggregate_cluster(current_cluster))
        current_cluster = [_copy_note(candidate)]
    clusters.append(aggregate_cluster(current_cluster))

    def has_matching_final_note(
        cluster_note: MelodyNote,
        snapped_position: float,
        raw_position: float,
        repeated_context: bool,
    ) -> bool:
        for note in recovered:
            if abs(note.midi - cluster_note.midi) > 0:
                continue
            explicit_same_pitch_reattack = (
                repeated_context
                and cluster_note.onset_supported
                and cluster_note.duration <= 0.18
                and note.start_time < cluster_note.start_time - 0.035
            )
            if explicit_same_pitch_reattack:
                continue
            if abs(note.start_time - cluster_note.start_time) <= 0.045:
                return True
            if abs(note.raw_beat_position - raw_position) <= 0.12:
                return True
            if (
                abs(note.quantized_beat_position - snapped_position) <= 0.12
                and abs(note.start_time - cluster_note.start_time) <= 0.08
            ):
                return True
        return False

    for index, cluster_note in enumerate(clusters):
        raw_position = (cluster_note.start_time - grid_origin_seconds) / beat_duration
        snapped_position = _snap_to_sixteenth_grid(raw_position)
        previous_cluster = clusters[index - 1] if index > 0 else None
        next_cluster = clusters[index + 1] if index + 1 < len(clusters) else None

        repeated_context = (
            (previous_cluster is not None and previous_cluster.midi == cluster_note.midi)
            or (next_cluster is not None and next_cluster.midi == cluster_note.midi)
        )
        contrast_context = (
            previous_cluster is not None
            and next_cluster is not None
            and previous_cluster.midi != cluster_note.midi
            and next_cluster.midi != cluster_note.midi
            and (
                previous_cluster.midi == next_cluster.midi
                or (
                    abs(cluster_note.midi - previous_cluster.midi) >= 2
                    and abs(cluster_note.midi - next_cluster.midi) >= 2
                )
            )
        )

        meaningful_short_cluster = (
            0.05 <= cluster_note.duration <= 0.18
            and (
                (
                    repeated_context
                    and (
                        cluster_note.onset_supported
                        or cluster_note.onset_strength >= 0.24
                        or cluster_note.confidence >= 0.82
                    )
                )
                or (
                    contrast_context
                    and (
                        cluster_note.confidence >= 0.62
                        or cluster_note.onset_supported
                        or cluster_note.onset_strength >= 0.12
                    )
                )
            )
        )
        if not meaningful_short_cluster:
            continue
        if has_matching_final_note(
            cluster_note,
            snapped_position,
            raw_position,
            repeated_context,
        ):
            continue

        inserted = _copy_note(cluster_note)
        inserted.quantized_beat_position = round(snapped_position, 4)
        inserted.quantized_duration_beats = round(max(0.25, _quantize_duration_to_grid(cluster_note.duration / beat_duration)), 4)
        inserted.measure_number = max(1, int(np.floor(snapped_position / beats_per_measure)) + 1)
        inserted.beat_in_measure = round((snapped_position % beats_per_measure) + 1, 4)
        inserted.rhythm_value = _rhythm_value_from_duration(inserted.quantized_duration_beats)
        inserted.beam_group = None
        inserted.beat_pattern = None
        inserted.tuplet_group = None
        recovered.append(inserted)

    recovered = sorted(recovered, key=lambda note: (note.quantized_beat_position, note.start_time, note.midi))
    for note_index, note in enumerate(recovered, start=1):
        note.note_index = note_index
    return recovered


def _repair_adjacent_isolated_dotted_pairs(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    repaired = [_copy_note(note) for note in notes]
    for index in range(len(repaired) - 1):
        first = repaired[index]
        second = repaired[index + 1]

        if first.beat_pattern in {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}:
            continue
        if second.beat_pattern in {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}:
            continue

        raw_diff = second.raw_beat_position - first.raw_beat_position
        prev_diff = None if index == 0 else first.raw_beat_position - repaired[index - 1].raw_beat_position
        next_diff = None if index + 2 >= len(repaired) else repaired[index + 2].raw_beat_position - second.raw_beat_position
        isolated_pair = (
            (prev_diff is None or prev_diff >= 0.42)
            and (next_diff is None or next_diff >= 0.4)
        )
        if not isolated_pair:
            continue

        strong_late_tail = (
            0.58 <= raw_diff <= 0.95
            and second.duration <= 0.18
            and first.duration >= max(0.18, second.duration * 1.35)
            and (
                second.onset_supported
                or second.onset_strength >= 0.28
                or second.confidence >= 0.88
            )
        )
        strong_early_tail = (
            0.15 <= raw_diff <= 0.38
            and first.duration <= 0.18
            and second.duration >= max(0.18, first.duration * 1.35)
            and (
                first.onset_supported
                or first.onset_strength >= 0.28
                or first.confidence >= 0.88
            )
        )

        if strong_late_tail:
            first_position = round(first.quantized_beat_position, 4)
            _set_note_quantization(first, first_position, 0.75, beats_per_measure)
            _set_note_quantization(second, round(first_position + 0.75, 4), 0.25, beats_per_measure)
            continue

        if strong_early_tail:
            first_position = round(first.quantized_beat_position, 4)
            _set_note_quantization(first, first_position, 0.25, beats_per_measure)
            _set_note_quantization(second, round(first_position + 0.25, 4), 0.75, beats_per_measure)

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _repair_contextual_dotted_pairs(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    repaired = [_copy_note(note) for note in notes]
    dotted_patterns = {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}
    weak_patterns = {
        None,
        "four_sixteenths",
        "two_eighths",
        "front_sixteenth_then_eighth",
        "front_eighth_then_sixteenth",
        "quarter",
    }

    for index in range(len(repaired) - 1):
        first = repaired[index]
        second = repaired[index + 1]
        if first.beat_pattern in dotted_patterns or second.beat_pattern in dotted_patterns:
            continue
        if first.tuplet_group or second.tuplet_group:
            continue
        if first.beat_pattern not in weak_patterns or second.beat_pattern not in weak_patterns:
            continue

        raw_diff = second.raw_beat_position - first.raw_beat_position
        if raw_diff <= 0:
            continue

        previous_note = repaired[index - 1] if index > 0 else None
        next_note = repaired[index + 2] if index + 2 < len(repaired) else None
        previous_gap = None if previous_note is None else first.raw_beat_position - previous_note.raw_beat_position
        next_gap = None if next_note is None else next_note.raw_beat_position - second.raw_beat_position

        late_tail = (
            0.6 <= raw_diff <= 0.92
            and second.duration <= 0.16
            and first.duration >= max(0.18, second.duration * 1.45)
            and (
                second.onset_supported
                or second.onset_strength >= 0.24
                or second.confidence >= 0.84
            )
        )
        early_tail = (
            0.18 <= raw_diff <= 0.34
            and first.duration <= 0.16
            and second.duration >= max(0.18, first.duration * 1.45)
            and (
                first.onset_supported
                or first.onset_strength >= 0.24
                or first.confidence >= 0.84
            )
        )

        if late_tail:
            # Keep this local pair together unless another note lands almost on top of the short tail.
            if next_gap is not None and next_gap < 0.12:
                continue
            base_position = round(first.quantized_beat_position * 4) / 4
            _set_note_quantization(first, base_position, 0.75, beats_per_measure)
            _set_note_quantization(second, round(base_position + 0.75, 4), 0.25, beats_per_measure)
            continue

        if early_tail:
            # Prefer a short pickup into a longer anchor only when the pair is locally coherent.
            if previous_gap is not None and previous_gap < 0.12:
                continue
            base_position = round(first.quantized_beat_position * 4) / 4
            _set_note_quantization(first, base_position, 0.25, beats_per_measure)
            _set_note_quantization(second, round(base_position + 0.25, 4), 0.75, beats_per_measure)

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _has_reliable_dotted_anchors(notes: list[MelodyNote]) -> bool:
    if len(notes) < 2:
        return False

    dotted_patterns = {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}
    reliable_anchor_count = 0

    for index in range(len(notes) - 1):
        first = notes[index]
        second = notes[index + 1]
        raw_diff = second.raw_beat_position - first.raw_beat_position

        explicit_dotted_pair = (
            first.beam_group is not None
            and first.beam_group == second.beam_group
            and first.beat_pattern in dotted_patterns
            and second.beat_pattern == first.beat_pattern
        )

        same_pitch_dotted_tail = (
            first.midi == second.midi
            and 0.58 <= raw_diff <= 0.92
            and first.duration >= 0.18
            and second.duration <= 0.18
            and (
                second.onset_supported
                or second.onset_strength >= 0.24
                or second.confidence >= 0.84
            )
        )

        same_pitch_dotted_head = (
            first.midi == second.midi
            and 0.18 <= raw_diff <= 0.34
            and first.duration <= 0.18
            and second.duration >= 0.18
            and (
                first.onset_supported
                or first.onset_strength >= 0.24
                or first.confidence >= 0.84
            )
        )

        if explicit_dotted_pair or same_pitch_dotted_tail or same_pitch_dotted_head:
            reliable_anchor_count += 1
            if reliable_anchor_count >= 1:
                return True

    return False


def _suppress_unreliable_dotted_patterns(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    repaired = [_copy_note(note) for note in notes]
    dotted_patterns = {"dotted_eighth_then_sixteenth", "sixteenth_then_dotted_eighth"}
    groups: dict[str, list[MelodyNote]] = {}
    for note in repaired:
        if note.beam_group and note.beat_pattern in dotted_patterns:
            groups.setdefault(note.beam_group, []).append(note)

    for group_notes in groups.values():
        ordered = sorted(group_notes, key=lambda item: item.quantized_beat_position)
        if len(ordered) != 2:
            continue

        base_position = round(min(note.quantized_beat_position for note in ordered) * 2) / 2
        _set_note_quantization(ordered[0], base_position, 0.5, beats_per_measure)
        _set_note_quantization(ordered[1], round(base_position + 0.5, 4), 0.5, beats_per_measure)
        ordered[0].rhythm_value = "eighth"
        ordered[1].rhythm_value = "eighth"
        ordered[0].beat_pattern = "two_eighths"
        ordered[1].beat_pattern = "two_eighths"

    for note_index, note in enumerate(repaired, start=1):
        note.note_index = note_index
    return repaired


def _repair_triplet_anchor_sequences(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 4:
        return notes

    repaired = [_copy_note(note) for note in notes]
    index = 0
    while index + 3 < len(repaired):
        first, second, third = repaired[index], repaired[index + 1], repaired[index + 2]
        if not (
            first.midi == second.midi == third.midi
            and abs(first.quantized_duration_beats - (1 / 3)) <= 0.02
            and abs(second.quantized_duration_beats - (1 / 3)) <= 0.02
            and abs(third.quantized_duration_beats - (1 / 3)) <= 0.02
        ):
            index += 1
            continue

        next_note = repaired[index + 3]
        triplet_beat_start = round(first.quantized_beat_position)
        expected_triplet = [triplet_beat_start, triplet_beat_start + 1 / 3, triplet_beat_start + 2 / 3]
        actual_triplet = [first.quantized_beat_position, second.quantized_beat_position, third.quantized_beat_position]
        if not all(abs(actual - expected) <= 0.2 for actual, expected in zip(actual_triplet, expected_triplet)):
            index += 1
            continue

        if next_note.midi != third.midi:
            anchor_position = triplet_beat_start + 1.0
            if abs(next_note.quantized_beat_position - anchor_position) <= 0.8:
                next_note.quantized_beat_position = round(anchor_position, 4)
                next_note.quantized_duration_beats = max(1.0, next_note.quantized_duration_beats)
                next_note.measure_number = max(1, int(np.floor(anchor_position / beats_per_measure)) + 1)
                next_note.beat_in_measure = round((anchor_position % beats_per_measure) + 1, 4)

                if index + 4 < len(repaired) and repaired[index + 4].midi == next_note.midi:
                    second_anchor = repaired[index + 4]
                    second_anchor_position = anchor_position + 1.0
                    if abs(second_anchor.quantized_beat_position - second_anchor_position) <= 0.9:
                        second_anchor.quantized_beat_position = round(second_anchor_position, 4)
                        second_anchor.quantized_duration_beats = max(1.0, second_anchor.quantized_duration_beats)
                        second_anchor.measure_number = max(1, int(np.floor(second_anchor_position / beats_per_measure)) + 1)
                        second_anchor.beat_in_measure = round((second_anchor_position % beats_per_measure) + 1, 4)

        index += 3

    return repaired


def _repair_double_triplet_anchor_sequences(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 9:
        return notes

    repaired = [_copy_note(note) for note in notes]
    index = 0
    while index + 8 < len(repaired):
        run = repaired[index : index + 6]
        run_pitch = run[0].midi
        base_beat = round(run[0].quantized_beat_position)
        expected_positions = [
            base_beat,
            base_beat + 1 / 3,
            base_beat + 2 / 3,
            base_beat + 1.0,
            base_beat + 4 / 3,
            base_beat + 5 / 3,
        ]

        if not all(
            note.midi == run_pitch and abs(note.quantized_duration_beats - (1 / 3)) <= 0.03
            for note in run
        ):
            index += 1
            continue

        if not all(abs(note.quantized_beat_position - expected) <= 0.2 for note, expected in zip(run, expected_positions)):
            index += 1
            continue

        first_anchor = repaired[index + 6]
        second_anchor = repaired[index + 7]
        next_phrase_note = repaired[index + 8]

        if first_anchor.midi == run_pitch or second_anchor.midi != first_anchor.midi:
            index += 1
            continue

        anchor_spacing = second_anchor.raw_beat_position - first_anchor.raw_beat_position
        next_spacing = next_phrase_note.raw_beat_position - second_anchor.raw_beat_position
        if not (0.75 <= anchor_spacing <= 1.25 and 0.75 <= next_spacing <= 1.25):
            index += 1
            continue

        first_anchor_position = base_beat + 2.0
        second_anchor_position = base_beat + 3.0
        next_phrase_position = base_beat + 4.0

        for anchor_note, target_position in (
            (first_anchor, first_anchor_position),
            (second_anchor, second_anchor_position),
        ):
            anchor_note.quantized_beat_position = round(target_position, 4)
            anchor_note.quantized_duration_beats = 1.0
            anchor_note.measure_number = max(1, int(np.floor(target_position / beats_per_measure)) + 1)
            anchor_note.beat_in_measure = round((target_position % beats_per_measure) + 1, 4)

        if abs(next_phrase_note.quantized_beat_position - next_phrase_position) <= 0.8:
            next_phrase_note.quantized_beat_position = round(next_phrase_position, 4)
            next_phrase_note.measure_number = max(1, int(np.floor(next_phrase_position / beats_per_measure)) + 1)
            next_phrase_note.beat_in_measure = round((next_phrase_position % beats_per_measure) + 1, 4)

        index += 6

    return repaired


def _rebuild_measure_grid(
    bpm_estimate: float,
    beats_per_measure: int,
    grid_origin_seconds: float,
    visible_duration_seconds: float,
) -> tuple[list[float], list[float]]:
    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
    strong_beats: list[float] = []
    measure_boundaries: list[float] = []
    current_boundary = grid_origin_seconds
    while current_boundary + beat_duration * beats_per_measure < 0:
        current_boundary += beat_duration * beats_per_measure
    while current_boundary <= visible_duration_seconds + beat_duration:
        rounded_boundary = round(current_boundary, 4)
        measure_boundaries.append(rounded_boundary)
        strong_beats.append(rounded_boundary)
        current_boundary += beat_duration * beats_per_measure
    if not measure_boundaries:
        return [0.0], [0.0]
    return strong_beats, measure_boundaries


def _duration_symbolic_fit_score(notes: list[MelodyNote], beat_duration: float) -> tuple[float, float]:
    if not notes or beat_duration <= 0:
        return 1.0, 0.0

    symbolic_targets = [0.25, 1 / 3, 0.5, 2 / 3, 0.75, 1.0, 1.5, 2.0]
    errors: list[float] = []
    matched = 0
    for note in notes:
        duration_beats = note.duration / beat_duration
        error = min(abs(duration_beats - target) for target in symbolic_targets)
        errors.append(error)
        if error <= 0.12:
            matched += 1
    mean_error = float(np.mean(errors)) if errors else 1.0
    match_ratio = matched / max(len(notes), 1)
    return mean_error, match_ratio


def _decode_rhythm_hypothesis_for_scoring(
    notes: list[MelodyNote],
    beat_duration: float,
    beats_per_measure: int,
    origin: float,
) -> tuple[list[MelodyNote], list[float], list[float]]:
    raw_positions = [(note.start_time - origin) / beat_duration for note in notes]
    quantized_positions = [_snap_to_sixteenth_grid(position) for position in raw_positions]
    hypothesis_notes = _build_hypothesis_notes(
        notes,
        quantized_positions,
        beat_duration,
        beats_per_measure,
    )
    hypothesis_notes = _apply_triplet_aware_quantization(
        hypothesis_notes,
        beats_per_measure=beats_per_measure,
    )
    hypothesis_notes = _resolve_non_triplet_subbeat_patterns(
        hypothesis_notes,
        beats_per_measure=beats_per_measure,
        allow_dotted=False,
    )
    hypothesis_notes = _repair_boundary_reactivation_artifacts(
        hypothesis_notes,
        beats_per_measure=beats_per_measure,
    )
    hypothesis_notes = _resolve_monophonic_mixed_subdivision_collisions(
        hypothesis_notes,
        beats_per_measure=beats_per_measure,
    )
    hypothesis_notes = _annotate_rhythm_groups(hypothesis_notes)
    return hypothesis_notes, raw_positions, quantized_positions


def _decoded_rhythm_structure_score(
    decoded_notes: list[MelodyNote],
    raw_positions: list[float],
) -> tuple[float, float, float]:
    if not decoded_notes:
        return 0.0, 0.0, 0.0

    beamed_ratio = sum(1 for note in decoded_notes if note.beam_group) / max(len(decoded_notes), 1)
    mixed_pattern_ratio = (
        sum(
            1
            for note in decoded_notes
            if note.beat_pattern in {
                "dotted_eighth_then_sixteenth",
                "sixteenth_then_dotted_eighth",
                "front_sixteenth_then_eighth",
                "front_eighth_then_sixteenth",
                "four_sixteenths",
                "triplet_eighths",
            }
        )
        / max(len(decoded_notes), 1)
    )

    raw_diffs = [
        round(curr - prev, 4)
        for prev, curr in zip(raw_positions, raw_positions[1:])
    ]
    fast_motion_ratio = (
        sum(diff <= 0.75 for diff in raw_diffs) / max(len(raw_diffs), 1)
        if raw_diffs
        else 0.0
    )

    subdivision_penalty = 0.0
    if fast_motion_ratio >= 0.42 and beamed_ratio <= 0.24:
        subdivision_penalty += 0.32
    if fast_motion_ratio >= 0.30 and mixed_pattern_ratio == 0.0 and beamed_ratio <= 0.35:
        subdivision_penalty += 0.14

    count_variability_penalty = 0.0
    measure_counts: dict[int, int] = {}
    for note in decoded_notes:
        measure_counts[note.measure_number] = measure_counts.get(note.measure_number, 0) + 1
    if len(measure_counts) >= 4:
        counts = np.asarray(list(measure_counts.values()), dtype=float)
        median_count = float(np.median(counts))
        mean_abs_deviation = float(np.mean(np.abs(counts - median_count)))
        if mean_abs_deviation >= 2.8:
            count_variability_penalty += 0.18
        elif mean_abs_deviation >= 2.1:
            count_variability_penalty += 0.08

    structure_bonus = beamed_ratio * 0.18 + mixed_pattern_ratio * 0.12
    return structure_bonus, subdivision_penalty, count_variability_penalty


def _refine_rhythm_with_segmented_notes(
    rhythm: RhythmAnalysis,
    notes: list[MelodyNote],
    visible_duration_seconds: float,
) -> RhythmAnalysis:
    if len(notes) < 4:
        return rhythm

    bpm_candidates: list[float] = [rhythm.bpm_estimate]
    candidate_values = [
        rhythm.bpm_estimate / 2.0,
        rhythm.bpm_estimate * 2.0,
    ]
    if rhythm.beats_per_measure == 3 or "3/4" in rhythm.time_signature_tendency:
        candidate_values.extend(
            [
                rhythm.bpm_estimate * (2.0 / 3.0),
                rhythm.bpm_estimate * 1.5,
            ]
        )

    for candidate in candidate_values:
        if 45.0 <= candidate <= 220.0 and all(abs(candidate - existing) > 0.01 for existing in bpm_candidates):
            bpm_candidates.append(candidate)

    candidate_meters = [3, 4]
    note_starts = [note.start_time for note in notes]
    earliest_note_start = min(note_starts)
    best_meter = rhythm.beats_per_measure
    best_origin = rhythm.grid_origin_seconds
    best_bpm = rhythm.bpm_estimate
    best_score = float("-inf")
    second_best_score = float("-inf")

    for bpm_estimate in bpm_candidates:
        beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
        latest_reasonable_origin = earliest_note_start + beat_duration * 0.35
        earliest_reasonable_origin = earliest_note_start - beat_duration * 1.5

        for beats_per_measure in candidate_meters:
            candidate_origins = [rhythm.grid_origin_seconds]
            for phase in range(beats_per_measure):
                for subphase in (0.0, 0.25, 0.5, 0.75):
                    for note_start in note_starts[: min(len(note_starts), 16)]:
                        candidate_origin = note_start - (phase + subphase) * beat_duration
                        if earliest_reasonable_origin <= candidate_origin <= latest_reasonable_origin:
                            candidate_origins.append(candidate_origin)

            for origin in candidate_origins:
                hypothesis_notes, raw_positions, quantized_positions = _decode_rhythm_hypothesis_for_scoring(
                    notes,
                    beat_duration=beat_duration,
                    beats_per_measure=beats_per_measure,
                    origin=origin,
                )
                grid_errors = [
                    abs(raw_position - quantized_position)
                    for raw_position, quantized_position in zip(raw_positions, quantized_positions)
                ]
                average_grid_error = float(np.mean(grid_errors)) if grid_errors else 1.0
                if origin > latest_reasonable_origin or origin < earliest_reasonable_origin:
                    continue

                repeated_step_bonus = 0.0
                downbeat_bonus = 0.0
                consistent_measure_endings = 0
                pickup_bonus = 0.0
                oversubdivision_penalty = 0.0
                symbolic_duration_error, symbolic_match_ratio = _duration_symbolic_fit_score(
                    hypothesis_notes,
                    beat_duration,
                )
                repeated_phrase_bonus = _repeated_phrase_consistency_score(
                    hypothesis_notes,
                    beats_per_measure,
                )
                if bpm_estimate >= rhythm.bpm_estimate - 0.01:
                    repeated_phrase_bonus *= 0.35

                diffs = [round(curr - prev, 4) for prev, curr in zip(quantized_positions, quantized_positions[1:])]
                for previous_step, current_step in zip(diffs, diffs[1:]):
                    if abs(previous_step - current_step) <= 0.01:
                        repeated_step_bonus += 0.02
                sixteenth_step_count = sum(abs(step - 0.25) <= 0.01 for step in diffs)
                eighth_step_count = sum(abs(step - 0.5) <= 0.01 for step in diffs)
                quarter_step_count = sum(abs(step - 1.0) <= 0.01 for step in diffs)
                if bpm_estimate < rhythm.bpm_estimate:
                    # If slowing the beat level causes quarter-beat steps to dominate,
                    # the system is probably over-subdividing the melody rather than
                    # discovering a truer metrical level.
                    if (
                        sixteenth_step_count >= 6
                        and sixteenth_step_count >= eighth_step_count + 4
                        and sixteenth_step_count > quarter_step_count * 1.5
                    ):
                        oversubdivision_penalty += 0.55
                    elif (
                        sixteenth_step_count > eighth_step_count
                        and sixteenth_step_count > quarter_step_count
                    ):
                        oversubdivision_penalty += 0.22

                duration_beats = [note.duration / beat_duration for note in notes]
                for index, (note, quantized_position) in enumerate(zip(hypothesis_notes, quantized_positions)):
                    beat_in_measure = (quantized_position % beats_per_measure) + 1
                    if abs(beat_in_measure - round(beat_in_measure)) <= 0.01:
                        downbeat_bonus += 0.005
                    if note.duration >= beat_duration * 0.35 and beat_in_measure >= beats_per_measure - 0.01:
                        consistent_measure_endings += 1

                for index in range(len(notes) - 1):
                    current_position = raw_positions[index]
                    next_position = raw_positions[index + 1]
                    current_duration = duration_beats[index]
                    next_duration = duration_beats[index + 1]
                    spacing = next_position - current_position
                    current_fraction = current_position - np.floor(current_position)
                    next_fraction = next_position - np.floor(next_position)
                    short_pickup = 0.12 <= current_duration <= 0.45
                    longer_follow = next_duration >= 0.35
                    close_follow = 0.18 <= spacing <= 0.38
                    aligned_follow = (
                        abs(next_fraction - 0.0) <= 0.12
                        or abs(next_fraction - 0.25) <= 0.12
                        or abs(next_fraction - 0.5) <= 0.12
                    )
                    if short_pickup and longer_follow and close_follow and aligned_follow:
                        pickup_bonus += 0.06
                        if abs(current_fraction - 0.75) <= 0.14 or abs(current_fraction - 0.0) <= 0.14:
                            pickup_bonus += 0.03

                common_time_bias = 0.12 if beats_per_measure == 4 else 0.0
                half_tempo_bias = 0.08 if bpm_estimate < rhythm.bpm_estimate else 0.0
                negative_position_penalty = sum(position < -0.5 for position in quantized_positions) * 0.03
                structure_bonus, structure_penalty, count_variability_penalty = _decoded_rhythm_structure_score(
                    hypothesis_notes,
                    raw_positions,
                )
                score = (
                    -average_grid_error * 3.2
                    -symbolic_duration_error * 3.8
                    +symbolic_match_ratio * 0.4
                    + repeated_step_bonus
                    + downbeat_bonus
                    + consistent_measure_endings * 0.015
                    + pickup_bonus
                    + structure_bonus
                    + repeated_phrase_bonus * 0.55
                    + common_time_bias
                    + half_tempo_bias
                    - oversubdivision_penalty
                    - structure_penalty
                    - count_variability_penalty
                    - negative_position_penalty
                )
                if score > best_score:
                    second_best_score = best_score
                    best_score = score
                    best_meter = beats_per_measure
                    best_origin = origin
                    best_bpm = bpm_estimate
                elif score > second_best_score:
                    second_best_score = score

    ambiguity_note = ""
    if second_best_score > float("-inf") and (best_score - second_best_score) <= 0.14:
        ambiguity_note = " Rhythm hypothesis margin is small; treat BPM and beat placement as lower confidence."

    if (
        best_meter == rhythm.beats_per_measure
        and abs(best_origin - rhythm.grid_origin_seconds) <= 0.01
        and abs(best_bpm - rhythm.bpm_estimate) <= 0.01
    ):
        return rhythm

    beat_duration = 60.0 / best_bpm if best_bpm > 0 else 60.0 / 90.0
    beat_positions: list[float] = []
    current_beat = best_origin
    while current_beat + beat_duration < 0:
        current_beat += beat_duration
    while current_beat <= visible_duration_seconds + beat_duration:
        beat_positions.append(round(current_beat, 4))
        current_beat += beat_duration

    strong_beats, measure_boundaries = _rebuild_measure_grid(
        bpm_estimate=best_bpm,
        beats_per_measure=best_meter,
        grid_origin_seconds=round(best_origin, 4),
        visible_duration_seconds=visible_duration_seconds,
    )
    return RhythmAnalysis(
        bpm_estimate=round(best_bpm, 2),
        bpm_confidence_note=f"{rhythm.bpm_confidence_note} Refined with multi-hypothesis note-grid fitting.{ambiguity_note}",
        beat_positions=beat_positions,
        strong_beat_positions=strong_beats,
        measure_boundaries=measure_boundaries,
        beats_per_measure=best_meter,
        grid_origin_seconds=round(best_origin, 4),
        quantization_grid=rhythm.quantization_grid,
        time_signature_tendency=f"{best_meter}/4 tendency",
    )


def _regularize_repeated_note_rhythm(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 3:
        return notes

    regularized = [_copy_note(note) for note in notes]
    run_start = 0

    def apply_run(start: int, end: int) -> None:
        run = regularized[start:end]
        if len(run) < 3:
            return
        if len({note.midi for note in run}) != 1:
            return
        if any(abs(note.quantized_duration_beats - (1 / 3)) <= 0.02 for note in run):
            return
        durations = [note.quantized_duration_beats for note in run]
        if max(durations) - min(durations) > 0.5:
            return
        median_duration = _median_or_default(durations, 0.5)
        target_duration = _quantize_duration_to_grid(median_duration)
        starts = [note.quantized_beat_position for note in run]
        diffs = [round(curr - prev, 4) for prev, curr in zip(starts, starts[1:])]
        if not diffs:
            return
        median_diff = _median_or_default(diffs, target_duration)
        sixteenth_like = sum(diff <= 0.3 for diff in diffs) >= max(2, len(diffs) // 2)
        if sixteenth_like:
            target_step = 0.25
        else:
            target_step = 0.5 if median_diff <= 0.75 else _quantize_duration_to_grid(median_diff)
        if target_step > 1.0:
            return
        target_duration = 0.25 if sixteenth_like else max(0.5, target_duration)
        for offset, note in enumerate(run):
            snapped_position = starts[0] + offset * target_step
            note.quantized_beat_position = round(_snap_to_sixteenth_grid(snapped_position), 4)
            note.quantized_duration_beats = round(target_duration, 4)
            note.measure_number = max(1, int(np.floor(note.quantized_beat_position / beats_per_measure)) + 1)
            note.beat_in_measure = round((note.quantized_beat_position % beats_per_measure) + 1, 4)

    for index in range(1, len(regularized)):
        previous = regularized[index - 1]
        current = regularized[index]
        same_pitch = previous.midi == current.midi
        close_step = (current.quantized_beat_position - previous.quantized_beat_position) <= 1.0
        if not (same_pitch and close_step):
            apply_run(run_start, index)
            run_start = index
    apply_run(run_start, len(regularized))
    return regularized


def _refine_isolated_long_note_values(
    notes: list[MelodyNote],
    bpm_estimate: float,
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 1:
        return notes

    beat_duration = 60.0 / bpm_estimate if bpm_estimate > 0 else 60.0 / 90.0
    refined = [_copy_note(note) for note in notes]
    ordered = sorted(refined, key=lambda item: item.quantized_beat_position)
    allowed_values = [1.0, 1.5, 2.0, 3.0, 4.0]

    for index, note in enumerate(ordered):
        previous_note = ordered[index - 1] if index > 0 else None
        next_note = ordered[index + 1] if index + 1 < len(ordered) else None

        if previous_note is not None and abs(note.quantized_beat_position - previous_note.quantized_beat_position) < 0.01:
            continue
        if next_note is not None and abs(next_note.quantized_beat_position - note.quantized_beat_position) < 0.01:
            continue

        raw_duration_beats = note.duration / beat_duration if beat_duration > 0 else note.quantized_duration_beats
        if raw_duration_beats < 1.2:
            continue

        next_start = next_note.quantized_beat_position if next_note is not None else (
            np.floor(note.quantized_beat_position / beats_per_measure) * beats_per_measure + beats_per_measure
        )
        available_span = max(0.25, next_start - note.quantized_beat_position)
        if available_span < 1.2:
            continue

        candidate_values = [value for value in allowed_values if value <= available_span + 0.08]
        if not candidate_values:
            continue

        best_value = min(candidate_values, key=lambda value: abs(value - raw_duration_beats))
        if best_value >= note.quantized_duration_beats + 0.25:
            note.quantized_duration_beats = round(best_value, 4)

    return refined


def _annotate_rhythm_groups(notes: list[MelodyNote]) -> list[MelodyNote]:
    if not notes:
        return notes

    annotated = [_copy_note(note) for note in notes]
    grouped_by_measure: dict[int, list[MelodyNote]] = {}
    for note in annotated:
        note.beam_group = None
        grouped_by_measure.setdefault(note.measure_number, []).append(note)

    for measure_notes in grouped_by_measure.values():
        sorted_notes = sorted(measure_notes, key=lambda item: item.quantized_beat_position)
        for index, note in enumerate(sorted_notes):
            note.beam_group = None
            note.tuplet_group = None
            note.beat_pattern = None
            previous_step = None
            next_step = None
            if index > 0:
                previous_step = round(note.quantized_beat_position - sorted_notes[index - 1].quantized_beat_position, 4)
            if index + 1 < len(sorted_notes):
                next_step = round(sorted_notes[index + 1].quantized_beat_position - note.quantized_beat_position, 4)
            note.rhythm_value = _classify_rhythm_value(note.quantized_duration_beats, previous_step, next_step)

        beam_counter = 1
        tuplet_counter = 1
        preassigned: set[int] = set()
        for index in range(len(sorted_notes) - 1):
            first = sorted_notes[index]
            second = sorted_notes[index + 1]
            if first.note_index in preassigned or second.note_index in preassigned:
                continue

            is_existing_dotted_pair = (
                abs(first.quantized_duration_beats - 0.75) <= 0.08
                and second.quantized_duration_beats <= 0.26
                and abs((second.quantized_beat_position - first.quantized_beat_position) - 0.75) <= 0.08
            ) or (
                first.quantized_duration_beats <= 0.26
                and abs(second.quantized_duration_beats - 0.75) <= 0.08
                and abs((second.quantized_beat_position - first.quantized_beat_position) - 0.25) <= 0.08
            )

            is_same_pitch_syncopated_dotted_pair = (
                first.midi == second.midi
                and 0.45 <= first.quantized_duration_beats <= 0.75
                and second.quantized_duration_beats <= 0.26
                and abs((second.quantized_beat_position - first.quantized_beat_position) - 0.75) <= 0.08
                and second.start_time - first.end_time <= 0.06
                and (
                    second.onset_supported
                    or second.onset_strength >= 0.32
                    or second.confidence >= 0.9
                )
            )
            # The first annotation pass should stay conservative. If we label
            # ordinary two-note melodic motion as dotted too early, later
            # stages treat the whole piece as "dotted-friendly" and case2-like
            # straight rhythms collapse into dotted pairs. Contextual dotted
            # repairs still happen later, but only after stronger evidence is
            # established.
            is_contextual_late_dotted_pair = False
            is_contextual_early_dotted_pair = False

            if (
                not is_existing_dotted_pair
                and not is_same_pitch_syncopated_dotted_pair
                and not is_contextual_late_dotted_pair
                and not is_contextual_early_dotted_pair
            ):
                continue

            beam_id = f"beam_{beam_counter}"
            beam_counter += 1
            if (
                abs((second.quantized_beat_position - first.quantized_beat_position) - 0.25) <= 0.08
                or is_contextual_early_dotted_pair
            ):
                first.beat_pattern = "sixteenth_then_dotted_eighth"
                second.beat_pattern = "sixteenth_then_dotted_eighth"
                first.rhythm_value = "sixteenth"
                second.rhythm_value = "dotted_eighth"
                first.quantized_duration_beats = 0.25
                second.quantized_duration_beats = 0.75
                if is_contextual_early_dotted_pair:
                    anchor = round(first.quantized_beat_position * 4) / 4
                    first.quantized_beat_position = round(anchor, 4)
                    second.quantized_beat_position = round(anchor + 0.25, 4)
                    second.measure_number = first.measure_number
                    second.beat_in_measure = round(first.beat_in_measure + 0.25, 4)
            else:
                first.beat_pattern = "dotted_eighth_then_sixteenth"
                second.beat_pattern = "dotted_eighth_then_sixteenth"
                first.rhythm_value = "dotted_eighth"
                second.rhythm_value = "sixteenth"
                first.quantized_duration_beats = 0.75
                second.quantized_duration_beats = 0.25
                if is_contextual_late_dotted_pair:
                    anchor = round(first.quantized_beat_position * 4) / 4
                    first.quantized_beat_position = round(anchor, 4)
                    second.quantized_beat_position = round(anchor + 0.75, 4)
                    second.measure_number = first.measure_number
                    second.beat_in_measure = round(first.beat_in_measure + 0.75, 4)
            first.beam_group = beam_id
            second.beam_group = beam_id
            first.tuplet_group = None
            second.tuplet_group = None
            preassigned.add(first.note_index)
            preassigned.add(second.note_index)

        beat_buckets: dict[int, list[MelodyNote]] = {}
        for note in sorted_notes:
            if note.note_index in preassigned:
                continue
            beat_bucket = int(np.floor(note.beat_in_measure - 1))
            beat_buckets.setdefault(beat_bucket, []).append(note)

        for beat_index in sorted(beat_buckets):
            bucket_notes = sorted(beat_buckets[beat_index], key=lambda item: item.quantized_beat_position)
            pattern_name = _recognize_beat_pattern(bucket_notes)
            beam_counter, tuplet_counter = _apply_beat_pattern(
                bucket_notes,
                pattern_name,
                beam_counter,
                tuplet_counter,
            )

    return annotated


def _normalize_measure_grid(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    normalized = sorted(
        (_copy_note(note) for note in notes),
        key=lambda item: (item.quantized_beat_position, item.start_time, item.note_index),
    )
    for note in normalized:
        _set_note_measure_and_beat(note, beats_per_measure)
    for note_index, note in enumerate(normalized, start=1):
        note.note_index = note_index
    return normalized


def _repair_illegal_barline_splits(
    notes: list[MelodyNote],
    beats_per_measure: int,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    repaired = _normalize_measure_grid(notes, beats_per_measure)
    merged: list[MelodyNote] = []
    index = 0
    while index < len(repaired):
        current = _copy_note(repaired[index])
        if index + 1 >= len(repaired):
            merged.append(current)
            index += 1
            continue

        candidate = repaired[index + 1]
        same_pitch = current.midi == candidate.midi
        boundary_split = (
            candidate.measure_number == current.measure_number + 1
            and current.beat_in_measure >= beats_per_measure - 0.5
            and candidate.beat_in_measure <= 1.25
        )
        no_new_attack = (not candidate.onset_supported) and candidate.onset_strength < 0.24
        tight_gap = (candidate.start_time - current.end_time) <= 0.08
        short_tail = candidate.duration <= max(0.18, current.duration * 0.55)

        if same_pitch and boundary_split and no_new_attack and tight_gap and short_tail:
            current.end_time = round(max(current.end_time, candidate.end_time), 4)
            current.duration = round(current.end_time - current.start_time, 4)
            current.offset_supported = candidate.offset_supported or current.offset_supported
            current.offset_strength = round(max(current.offset_strength, candidate.offset_strength), 4)
            merged.append(current)
            index += 2
            continue

        merged.append(current)
        index += 1

    return _normalize_measure_grid(merged, beats_per_measure)


def _post_correct_mixed_subdivision_short_pitches(
    notes: list[MelodyNote],
    allowed_midis: list[int] | None,
) -> list[MelodyNote]:
    if len(notes) < 2:
        return notes

    corrected = [_copy_note(note) for note in notes]
    candidate_universe = allowed_midis or UKULELE_PLAYABLE_MIDI
    beam_groups: dict[str, list[MelodyNote]] = {}
    for note in corrected:
        if note.beam_group:
            beam_groups.setdefault(note.beam_group, []).append(note)

    for group_notes in beam_groups.values():
        ordered = sorted(group_notes, key=lambda item: item.quantized_beat_position)
        pattern_name = ordered[0].beat_pattern
        if pattern_name not in {
            "front_sixteenth_then_eighth",
            "front_eighth_then_sixteenth",
            "dotted_eighth_then_sixteenth",
            "sixteenth_then_dotted_eighth",
        }:
            continue

        short_notes = [note for note in ordered if note.quantized_duration_beats <= 0.26]
        if not short_notes:
            continue

        local_midis = {note.midi for note in ordered}

        for short_note in short_notes:
            observed_frequency = short_note.observed_frequency_hz or short_note.frequency_hz
            if observed_frequency <= 0:
                continue
            observed_midi = float(librosa.hz_to_midi(observed_frequency))

            candidate_pool = [
                midi_candidate
                for midi_candidate in candidate_universe
                if abs(midi_candidate - observed_midi) <= 4.5 or midi_candidate in local_midis
            ]
            if not candidate_pool:
                continue

            current_index = ordered.index(short_note)
            previous_note = ordered[current_index - 1] if current_index > 0 else None
            next_note = ordered[current_index + 1] if current_index + 1 < len(ordered) else None

            def score_candidate(midi_candidate: int) -> float:
                score = abs(observed_midi - midi_candidate) * 1.35
                if previous_note is not None:
                    score += abs(midi_candidate - previous_note.midi) * 0.08
                if next_note is not None:
                    score += abs(next_note.midi - midi_candidate) * 0.08
                if previous_note is not None and next_note is not None and midi_candidate == previous_note.midi == next_note.midi:
                    score += 0.55
                if previous_note is not None and midi_candidate == previous_note.midi and short_note.midi != previous_note.midi:
                    score += 0.1
                if next_note is not None and midi_candidate == next_note.midi and short_note.midi != next_note.midi:
                    score += 0.1
                return score

            current_score = score_candidate(short_note.midi)
            best_candidate = min(candidate_pool, key=score_candidate)
            best_score = score_candidate(best_candidate)

            improvement_threshold = 0.45 if pattern_name in {
                "dotted_eighth_then_sixteenth",
                "sixteenth_then_dotted_eighth",
            } else 0.6
            if best_candidate != short_note.midi and best_score <= current_score - improvement_threshold:
                short_note.midi = int(best_candidate)
                short_note.note_name = _format_note_name(short_note.midi)
                short_note.frequency_hz = round(float(librosa.midi_to_hz(short_note.midi)), 3)

    return corrected


def _sanity_check_notes(notes: list[MelodyNote]) -> list[str]:
    warnings: list[str] = []
    if not notes:
        return ["No segmented notes were produced from the audio."]

    pitch_classes = {note.note_name[:-1] for note in notes if len(note.note_name) >= 2}
    if len(notes) >= 8 and len(pitch_classes) <= 2:
        warnings.append(
            f"Only {len(pitch_classes)} pitch classes found across {len(notes)} notes. Melody extraction may be collapsed."
        )

    repeated_nearby = 0
    for previous, current in zip(notes, notes[1:]):
        same_note = previous.midi == current.midi
        same_measure = previous.measure_number == current.measure_number
        same_beat_neighborhood = abs(previous.beat_in_measure - current.beat_in_measure) <= 0.5
        if same_note and same_measure and same_beat_neighborhood:
            repeated_nearby += 1
    if repeated_nearby >= 2:
        warnings.append(
            "Repeated identical notes are clustered in the same beat neighborhood. Segmentation may be over-splitting."
        )

    durations = [note.duration for note in notes]
    short_notes = sum(duration < 0.08 for duration in durations)
    if len(notes) >= 10 and short_notes / max(len(notes), 1) > 0.35:
        warnings.append(
            "Many segmented notes are extremely short. Note timeline may not match the original melody."
        )

    inferred_beats_per_measure = max(3, int(np.ceil(max(note.beat_in_measure for note in notes))))
    repeated_phrase_score = _repeated_phrase_consistency_score(notes, inferred_beats_per_measure)
    if len(notes) >= 12 and repeated_phrase_score <= 0.08:
        warnings.append(
            "Repeated phrase consistency is low. Similar melody sections may be decoding differently across the piece."
        )

    boundary_split_count = 0
    strong_offbeat_count = 0
    for previous, current in zip(notes, notes[1:]):
        if (
            previous.measure_number + 1 == current.measure_number
            and previous.midi == current.midi
            and current.beat_in_measure <= 1.25
            and previous.beat_in_measure >= inferred_beats_per_measure - 0.5
            and not current.onset_supported
            and current.onset_strength < 0.24
        ):
            boundary_split_count += 1

        if current.duration_beats >= 0.75:
            fractional = (current.beat_in_measure - 1) % 1.0
            if 0.24 <= fractional <= 0.76:
                strong_offbeat_count += 1

    if boundary_split_count >= 1:
        warnings.append(
            "Some same-pitch notes appear split across a measure boundary without clear re-attack evidence. Measure lines may be cutting one sustained sound into two notes."
        )
    if len(notes) >= 8 and strong_offbeat_count / max(len(notes), 1) > 0.2:
        warnings.append(
            "Many long notes are landing away from strong beats. BPM, beat phase, or measure boundaries may be misaligned."
        )

    return warnings


def _candidate_positions(midi_note: int) -> list[tuple[str, int, int]]:
    candidates: list[tuple[str, int, int]] = []
    for string_name, string_number, open_midi in UKULELE_STRINGS:
        fret = midi_note - open_midi
        if 0 <= fret <= 12:
            candidates.append((string_name, string_number, fret))
    return candidates


def _candidate_states(midi_note: int) -> list[tuple[str, int, int, int]]:
    states: list[tuple[str, int, int, int]] = []
    for string_name, string_number, fret in _candidate_positions(midi_note):
        if fret == 0:
            states.append((string_name, string_number, fret, 0))
            continue

        min_anchor = max(1, fret - 3)
        max_anchor = min(fret, 9)
        for anchor_fret in range(min_anchor, max_anchor + 1):
            states.append((string_name, string_number, fret, anchor_fret))

    if states:
        return states

    nearest = min(UKULELE_STRINGS, key=lambda item: abs(midi_note - item[2]))
    fallback_fret = max(0, min(12, midi_note - nearest[2]))
    fallback_anchor = 0 if fallback_fret == 0 else max(1, fallback_fret - 1)
    return [(nearest[0], nearest[1], fallback_fret, fallback_anchor)]


def _mapping_cost(
    prev_state: tuple[str, int, int, int] | None,
    current_state: tuple[str, int, int, int],
    midi_note: int,
) -> tuple[float, str]:
    string_name, string_number, fret, anchor_fret = current_state
    cost = 0.0
    reasons: list[str] = []

    if fret == 0:
        cost -= 0.35
        reasons.append("open string is easy")
    else:
        cost += fret * 0.45
        reasons.append(f"prefer manageable fret {fret}")

    if anchor_fret == 0:
        cost += 0.15
    else:
        cost += anchor_fret * 0.18
        reasons.append(f"hand position around fret {anchor_fret}")
        if fret - anchor_fret <= 2:
            cost -= 0.2
            reasons.append("note fits naturally in local hand shape")
        if fret - anchor_fret == 3:
            cost += 0.25
            reasons.append("full four-fret stretch")

    if string_name == "G":
        cost += 0.45
        reasons.append("re-entrant G string is less default for melody")
    elif string_name == "C":
        cost += 0.08
    elif string_name == "E":
        cost -= 0.05
    elif string_name == "A":
        cost -= 0.08
        reasons.append("top melody string is intuitive")

    if prev_state is not None:
        prev_string, prev_string_number, prev_fret, prev_anchor = prev_state
        fret_shift = abs(fret - prev_fret)
        string_shift = abs(string_number - prev_string_number)
        anchor_shift = abs(anchor_fret - prev_anchor)

        cost += fret_shift * 0.22
        cost += string_shift * 0.28
        cost += anchor_shift * 0.85
        reasons.append(f"anchor shift {anchor_shift}")

        if anchor_shift == 0 and anchor_fret != 0:
            cost -= 0.45
            reasons.append("stay in same hand position")
        elif anchor_shift == 1:
            cost -= 0.1
            reasons.append("small hand move")
        elif anchor_shift >= 4:
            cost += 2.4
            reasons.append("large position shift penalty")

        if prev_string == string_name and fret_shift <= 2:
            cost -= 0.18
            reasons.append("same string small move")
        elif string_shift == 1 and fret_shift <= 2:
            cost -= 0.08
            reasons.append("adjacent string within same shape")

        if prev_fret == 0 and fret >= 5:
            cost += 0.9
            reasons.append("avoid jumping from open string to high fret")
        if prev_fret >= 5 and fret == 0:
            cost += 0.45
            reasons.append("open string resets hand position")

    if midi_note >= 81:
        cost += 0.6
        reasons.append("high register caution")

    if string_name == "A" and fret >= 8:
        cost += 0.55
        reasons.append("high A-string fret is less comfortable")
    if string_name == "C" and 1 <= fret <= 5:
        cost -= 0.12
        reasons.append("middle strings often feel stable")
    if string_name == "E" and 1 <= fret <= 5:
        cost -= 0.1
        reasons.append("common melody area on E string")
    if string_name == "A" and fret == 0:
        cost -= 0.2
        reasons.append("open A is a natural melody starting point")

    return cost, ", ".join(reasons)


def map_notes_to_ukulele(notes: list[MelodyNote]) -> list[UkuleleMappedNote]:
    if not notes:
        return []

    dp: list[
        dict[tuple[str, int, int, int], tuple[float, tuple[str, int, int, int] | None, str]]
    ] = []

    for note_index, note in enumerate(notes):
        candidates = _candidate_states(note.midi)

        state_costs: dict[
            tuple[str, int, int, int],
            tuple[float, tuple[str, int, int, int] | None, str],
        ] = {}
        for candidate in candidates:
            if note_index == 0:
                candidate_cost, reason = _mapping_cost(None, candidate, note.midi)
                state_costs[candidate] = (candidate_cost, None, reason)
                continue

            best_total = float("inf")
            best_prev: tuple[str, int, int, int] | None = None
            best_reason = ""
            for prev_candidate, (prev_cost, _, _) in dp[-1].items():
                step_cost, reason = _mapping_cost(prev_candidate, candidate, note.midi)
                total_cost = prev_cost + step_cost
                if total_cost < best_total:
                    best_total = total_cost
                    best_prev = prev_candidate
                    best_reason = reason
            state_costs[candidate] = (best_total, best_prev, best_reason)

        dp.append(state_costs)

    end_state = min(dp[-1].items(), key=lambda item: item[1][0])[0]
    chosen_states: list[tuple[str, int, int, int]] = [end_state]
    for layer in range(len(dp) - 1, 0, -1):
        _, prev_state, _ = dp[layer][chosen_states[-1]]
        assert prev_state is not None
        chosen_states.append(prev_state)
    chosen_states.reverse()

    mapped_notes: list[UkuleleMappedNote] = []
    for note, state in zip(notes, chosen_states):
        string_name, string_number, fret, anchor_fret = state
        reason = dp[note.note_index - 1][state][2]
        mapped_notes.append(
            UkuleleMappedNote(
                note_index=note.note_index,
                note_name=note.note_name,
                midi=note.midi,
                start_time=note.start_time,
                duration=note.duration,
                measure_number=note.measure_number,
                beat_in_measure=note.beat_in_measure,
                quantized_duration_beats=note.quantized_duration_beats,
                chosen_string=string_name,
                chosen_string_number=string_number,
                chosen_fret=fret,
                hand_position_fret=anchor_fret,
                mapping_reason=reason,
            )
        )
    return mapped_notes


def render_tab_preview(mapped_notes: list[UkuleleMappedNote], beats_per_measure: int) -> str:
    if not mapped_notes:
        return "No mapped notes found."

    last_measure = max(note.measure_number for note in mapped_notes)
    rows: list[str] = []
    string_order = ["A", "E", "C", "G"]
    slots_per_measure = beats_per_measure * 4

    for measure_number in range(1, last_measure + 1):
        rows.append(f"Measure {measure_number}")
        measure_rows = {name: ["-"] * slots_per_measure for name in string_order}
        for note in [item for item in mapped_notes if item.measure_number == measure_number]:
            slot = min(
                slots_per_measure - 1,
                max(0, int(round((note.beat_in_measure - 1) * 4))),
            )
            fret_text = str(note.chosen_fret)
            for offset, char in enumerate(fret_text):
                if slot + offset < slots_per_measure:
                    measure_rows[note.chosen_string][slot + offset] = char
        for string_name in string_order:
            rows.append(f"{string_name}|{''.join(measure_rows[string_name])}|")
        rows.append("")

    return "\n".join(rows).strip()


def _draft_pdf_mapping_for_note(note: MelodyNote) -> tuple[str, int]:
    candidates = _candidate_positions(note.midi)
    if not candidates:
        return "A", 0

    string_preference = {"A": 0, "E": 1, "C": 2, "G": 3}
    best = min(
        candidates,
        key=lambda item: (item[2], string_preference.get(item[0], 9)),
    )
    return best[0], best[2]


def _render_draft_tab_pdf(
    output_path: str | Path,
    source_title: str,
    notes: list[MelodyNote],
    rhythm: RhythmAnalysis,
) -> str:
    if canvas is None or A4 is None:
        raise RuntimeError("reportlab is not installed")

    pdf_path = Path(output_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    page_width, page_height = A4
    pdf = canvas.Canvas(str(pdf_path), pagesize=A4)
    left_margin = 48
    right_margin = 48
    top_margin = 64
    line_gap = 12
    system_gap = 108
    usable_width = page_width - left_margin - right_margin
    measures_per_row = 4
    measure_width = usable_width / measures_per_row
    slots_per_measure = rhythm.beats_per_measure * 4
    slot_width = measure_width / max(slots_per_measure, 1)

    pdf.setTitle(f"{source_title} ukulele tab draft")
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(left_margin, page_height - top_margin, source_title)
    pdf.setFont("Helvetica", 10)
    pdf.drawString(left_margin, page_height - top_margin - 18, "auto-generated draft")
    pdf.drawString(
        left_margin,
        page_height - top_margin - 32,
        f"BPM {rhythm.bpm_estimate} | {rhythm.time_signature_tendency}",
    )

    measures: dict[int, list[MelodyNote]] = {}
    for note in notes:
        measures.setdefault(note.measure_number, []).append(note)
    last_measure = max(measures.keys(), default=1)

    def draw_rhythm_mark(
        pdf_canvas: canvas.Canvas,
        x_positions: list[float],
        rhythm_values: list[str],
        g_string_y: float,
        draw_triplet_label: bool = False,
    ) -> None:
        if not x_positions:
            return

        stem_top_y = g_string_y - 7
        primary_beam_y = g_string_y - 19
        secondary_beam_y = g_string_y - 23
        partial_ratio = 0.55
        dot_y = stem_top_y - 2

        for x in x_positions:
            pdf_canvas.line(x, primary_beam_y, x, stem_top_y)

        if len(x_positions) == 1:
            return

        pdf_canvas.setLineWidth(1.4)
        pdf_canvas.line(x_positions[0], primary_beam_y, x_positions[-1], primary_beam_y)

        beam_levels = [_beam_level_from_rhythm_value(value) for value in rhythm_values]
        for index, level in enumerate(beam_levels):
            if level < 2:
                continue

            prev_is_sixteenth = index > 0 and beam_levels[index - 1] >= 2
            next_is_sixteenth = index + 1 < len(beam_levels) and beam_levels[index + 1] >= 2

            if next_is_sixteenth:
                pdf_canvas.line(
                    x_positions[index],
                    secondary_beam_y,
                    x_positions[index + 1],
                    secondary_beam_y,
                )
                continue

            if prev_is_sixteenth:
                continue

            if index + 1 < len(x_positions):
                span = (x_positions[index + 1] - x_positions[index]) * partial_ratio
                pdf_canvas.line(
                    x_positions[index],
                    secondary_beam_y,
                    x_positions[index] + span,
                    secondary_beam_y,
                )
            elif index > 0:
                span = (x_positions[index] - x_positions[index - 1]) * partial_ratio
                pdf_canvas.line(
                    x_positions[index] - span,
                    secondary_beam_y,
                    x_positions[index],
                    secondary_beam_y,
                )

        pdf_canvas.setLineWidth(1.0)
        for x, rhythm_value in zip(x_positions, rhythm_values):
            if rhythm_value == "dotted_eighth":
                pdf_canvas.circle(x + 5, dot_y, 1.2, fill=1, stroke=0)
        if draw_triplet_label:
            center_x = (x_positions[0] + x_positions[-1]) / 2
            pdf_canvas.setFont("Helvetica-Bold", 8)
            pdf_canvas.drawCentredString(center_x, stem_top_y + 3, "3")

    current_y = page_height - top_margin - 80
    ordered_notes = sorted(notes, key=lambda item: (item.measure_number, item.beat_in_measure, item.start_time, item.note_index))
    for measure_group_start in range(1, last_measure + 1, measures_per_row):
        if current_y < 120:
            pdf.showPage()
            current_y = page_height - top_margin

        system_layouts: dict[int, tuple[float, float, str, int]] = {}
        for system_index in range(min(measures_per_row, last_measure - measure_group_start + 1)):
            measure_number = measure_group_start + system_index
            measure_x = left_margin + system_index * measure_width
            beam_segments: dict[str, list[tuple[float, str]]] = {}
            tuplet_segments: dict[str, str] = {}
            pdf.setFont("Helvetica", 8)
            pdf.drawString(measure_x, current_y + 18, str(measure_number))
            for string_offset, string_name in enumerate(PDF_STRING_ORDER):
                y = current_y - string_offset * line_gap
                pdf.line(measure_x, y, measure_x + measure_width, y)
                if system_index == 0:
                    pdf.drawString(measure_x - 14, y - 3, string_name)
            pdf.line(measure_x, current_y + 2, measure_x, current_y - 3 * line_gap - 2)
            pdf.line(measure_x + measure_width, current_y + 2, measure_x + measure_width, current_y - 3 * line_gap - 2)
            g_string_y = current_y - 3 * line_gap

            for note in measures.get(measure_number, []):
                beat_slot = int(round((note.beat_in_measure - 1) * 4))
                beat_slot = max(0, min(slots_per_measure - 1, beat_slot))
                slot_x = measure_x + beat_slot * slot_width + slot_width * 0.35
                string_name = note.string_name or _draft_pdf_mapping_for_note(note)[0]
                fret = note.fret_number if note.fret_number is not None else _draft_pdf_mapping_for_note(note)[1]
                string_y = current_y - PDF_STRING_ORDER.index(string_name) * line_gap
                pdf.setFillColorRGB(1, 1, 1)
                pdf.rect(slot_x - 4, string_y - 5, 12, 10, fill=1, stroke=0)
                pdf.setFillColorRGB(0, 0, 0)
                pdf.setFont("Helvetica", 9)
                fret_label = str(fret)
                if note.harmonic_candidate:
                    pdf.setFont("Helvetica-Oblique", 6)
                    pdf.drawCentredString(slot_x + 2, string_y + 9, "harm.")
                    pdf.setFont("Helvetica", 9)
                pdf.drawString(slot_x - 2, string_y - 3, fret_label)
                system_layouts[note.note_index] = (slot_x + 2, string_y, string_name, fret)
                if note.rhythm_value in {"quarter", "dotted_quarter", "half", "dotted_half", "whole"}:
                    quarter_top_y = g_string_y - 7
                    quarter_bottom_y = g_string_y - 23
                    pdf.line(slot_x + 2, quarter_bottom_y, slot_x + 2, quarter_top_y)
                    if note.rhythm_value in {"dotted_quarter", "dotted_half"}:
                        pdf.circle(slot_x + 7, quarter_top_y - 2, 1.2, fill=1, stroke=0)
                if note.beam_group:
                    beam_segments.setdefault(note.beam_group, []).append((slot_x + 2, note.rhythm_value))
                    if note.tuplet_group:
                        tuplet_segments[note.beam_group] = note.tuplet_group

            for beam_id, beam_notes in beam_segments.items():
                if len(beam_notes) < 2:
                    continue
                sorted_beam_notes = sorted(beam_notes, key=lambda item: item[0])
                draw_rhythm_mark(
                    pdf,
                    [x for x, _ in sorted_beam_notes],
                    [value for _, value in sorted_beam_notes],
                    g_string_y,
                    draw_triplet_label=beam_id in tuplet_segments,
                )

        pdf.setLineWidth(1.0)
        for current, following in zip(ordered_notes, ordered_notes[1:]):
            if not current.slide_to_next:
                continue
            if current.note_index not in system_layouts or following.note_index not in system_layouts:
                continue
            current_x, current_y_pos, current_string, _ = system_layouts[current.note_index]
            following_x, following_y_pos, following_string, _ = system_layouts[following.note_index]
            if current_string != following_string:
                continue
            pdf.line(current_x + 6, current_y_pos + 2, following_x - 4, following_y_pos + 2)

        current_y -= system_gap

    pdf.save()
    return str(pdf_path)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def write_intermediate_outputs(
    output_dir: str | Path,
    frame_pitch_debug: list[dict[str, object]],
    pitch_histogram: list[tuple[str, int]],
    source_classification: SourceClassification,
    rhythm: RhythmAnalysis,
    raw_notes: list[MelodyNote],
    quantized_notes: list[MelodyNote],
    notation_analysis: TabNotationAnalysis,
    core_inference: list[InferenceComponent],
    heuristic_corrections: list[InferenceComponent],
    validation_observations: list[ValidationObservation],
    segmentation_stats: dict[str, int | float | bool],
    tab_preview: str,
) -> dict[str, str]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    melody_debug_txt = out_dir / "melody_notes_debug.txt"
    pitch_histogram_txt = out_dir / "pitch_histogram.txt"
    segmented_notes_raw_json = out_dir / "segmented_notes_raw.json"
    segmented_notes_quantized_json = out_dir / "segmented_notes_quantized.json"
    source_classification_json = out_dir / "source_classification.json"
    notation_analysis_json = out_dir / "notation_analysis.json"
    inference_audit_json = out_dir / "inference_audit.json"
    segmentation_report_txt = out_dir / "segmentation_report.txt"
    mapping_debug_txt = out_dir / "ukulele_mapping_debug.txt"
    tab_txt = out_dir / "tab_preview.txt"

    _write_text(
        melody_debug_txt,
        "\n".join(
            [
                "Frame-level pitch estimates",
                *[
                    f"frame={frame['frame_index']:04d} time={frame['time']:.4f}s "
                    f"voiced={frame['voiced']} onset={frame['onset_frame']} "
                    f"freq={frame['frequency_hz']} midi={frame['midi']} "
                    f"note={frame['note_name']} conf={frame['confidence']}"
                    for frame in frame_pitch_debug
                ],
                "",
                "Segmented note-level timeline",
                *[
                    f"{note.note_index:03d} start={note.start_time:.4f}s end={note.end_time:.4f}s "
                    f"duration={note.duration:.4f}s note={note.note_name} midi={note.midi} "
                    f"conf={note.confidence:.4f}"
                    for note in raw_notes
                ],
            ]
        ),
    )
    _write_text(
        pitch_histogram_txt,
        "\n".join(
            [
                "Pitch histogram",
                *[f"{pitch_class}: {count}" for pitch_class, count in pitch_histogram],
            ]
        ),
    )
    _write_json(segmented_notes_raw_json, [note.to_dict() for note in raw_notes])
    _write_json(segmented_notes_quantized_json, [note.to_dict() for note in quantized_notes])
    _write_json(source_classification_json, source_classification.to_dict())
    _write_json(notation_analysis_json, notation_analysis.to_dict())
    _write_json(
        inference_audit_json,
        {
            "core_inference": [item.to_dict() for item in core_inference],
            "heuristic_corrections": [item.to_dict() for item in heuristic_corrections],
            "validation_observations": [item.to_dict() for item in validation_observations],
        },
    )
    _write_text(
        segmentation_report_txt,
        "\n".join(
            [
                "Segmentation report",
                f"raw note count: {segmentation_stats['raw_note_count']}",
                f"merged fragments: {segmentation_stats['merged_fragments']}",
                f"filtered short notes: {segmentation_stats['filtered_short_notes']}",
                f"final note count: {segmentation_stats['final_note_count']}",
                f"average note duration: {segmentation_stats['average_note_duration']}",
                f"dense repeat warning: {segmentation_stats['dense_repeat_warning']}",
                f"minimum duration threshold: 0.12",
                f"bpm estimate: {rhythm.bpm_estimate}",
                f"time signature tendency: {rhythm.time_signature_tendency}",
                f"beats per measure: {rhythm.beats_per_measure}",
                f"grid origin seconds: {rhythm.grid_origin_seconds}",
            ]
        ),
    )
    _write_text(
        mapping_debug_txt,
        "\n".join(
            [
                "Ukulele mapping debug",
                "status: skipped_until_segmented_notes_are_reviewed",
                f"bpm estimate: {rhythm.bpm_estimate}",
                f"time signature tendency: {rhythm.time_signature_tendency}",
                f"beat positions: {', '.join(f'{value:.4f}' for value in rhythm.beat_positions)}",
                f"measure segmentation: {', '.join(f'{value:.4f}' for value in rhythm.measure_boundaries)}",
                "Mapping intentionally disabled in this stage.",
            ]
        ),
    )
    _write_text(tab_txt, tab_preview)

    return {
        "melody_notes_debug_txt": str(melody_debug_txt),
        "pitch_histogram_txt": str(pitch_histogram_txt),
        "segmented_notes_raw_json": str(segmented_notes_raw_json),
        "segmented_notes_quantized_json": str(segmented_notes_quantized_json),
        "source_classification_json": str(source_classification_json),
        "notation_analysis_json": str(notation_analysis_json),
        "inference_audit_json": str(inference_audit_json),
        "segmentation_report_txt": str(segmentation_report_txt),
        "ukulele_mapping_debug_txt": str(mapping_debug_txt),
        "tab_preview_txt": str(tab_txt),
    }


def build_melody_pipeline(
    audio_path: str | Path,
    output_dir: str | Path | None = None,
    candidate_note_names: list[str] | None = None,
) -> dict[str, object]:
    path = Path(audio_path)
    out_dir = Path(output_dir) if output_dir else path.parent / f"{path.stem}_analysis"
    allowed_midis = [_parse_note_name(note_name) for note_name in candidate_note_names] if candidate_note_names else None

    print(f"preprocessing audio... {path}")
    audio, full_audio, preprocess = preprocess_audio(path, working_dir=out_dir)

    print("analyzing rhythm...")
    rhythm, beat_times = _estimate_beats(
        full_audio,
        preprocess.sample_rate,
        time_offset_seconds=preprocess.trimmed_start_seconds,
        visible_duration_seconds=preprocess.duration_seconds,
    )

    print("extracting frame-level pitch...")
    frame_pitch_debug, onset_frames = _extract_pitch_frames(audio, preprocess.sample_rate)
    voiced_midis = [int(frame["midi"]) for frame in frame_pitch_debug if frame["midi"] is not None]
    pitch_histogram = _build_pitch_histogram(voiced_midis)

    print("segmenting raw note-level melody...")
    raw_melody_notes = _segment_monophonic_notes(
        audio=audio,
        frame_pitch_debug=frame_pitch_debug,
        onset_frames=onset_frames,
        sample_rate=preprocess.sample_rate,
        allowed_midis=allowed_midis,
    )
    print("merging and filtering note fragments...")
    melody_notes, segmentation_stats = _merge_and_filter_notes(raw_melody_notes)
    melody_notes = _recover_high_register_bridge_notes(
        melody_notes,
        frame_pitch_debug=frame_pitch_debug,
        sample_rate=preprocess.sample_rate,
        allowed_midis=allowed_midis,
    )
    melody_notes = _optimize_pitch_path(
        melody_notes,
        allowed_midis=allowed_midis,
    )
    melody_notes = _contextual_short_note_pitch_correction(
        melody_notes,
        allowed_midis=allowed_midis,
    )
    rhythm_inference_notes = _notes_for_rhythm_inference(melody_notes)
    rhythm = _refine_rhythm_with_segmented_notes(
        rhythm,
        rhythm_inference_notes,
        visible_duration_seconds=preprocess.duration_seconds,
    )
    melody_notes = _consolidate_same_pitch_fragments(
        melody_notes,
        bpm_estimate=rhythm.bpm_estimate,
    )
    source_classification = _guess_source_classification(
        audio=audio,
        sample_rate=preprocess.sample_rate,
        frame_pitch_debug=frame_pitch_debug,
        notes=melody_notes,
    )
    segmentation_stats["final_note_count"] = len(melody_notes)
    segmentation_stats["average_note_duration"] = round(
        float(np.mean([note.duration for note in melody_notes])) if melody_notes else 0.0,
        4,
    )
    print("creating quantized note view...")
    quantized_melody_notes = _quantize_notes(
        melody_notes,
        bpm_estimate=rhythm.bpm_estimate,
        beats_per_measure=rhythm.beats_per_measure,
        grid_origin_seconds=rhythm.grid_origin_seconds,
    )
    quantized_melody_notes = _apply_triplet_aware_quantization(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _resolve_non_triplet_subbeat_patterns(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
        allow_dotted=False,
    )
    quantized_melody_notes = _repair_boundary_reactivation_artifacts(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _resolve_monophonic_mixed_subdivision_collisions(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _repair_split_dotted_subdivision_groups(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _repair_triplet_anchor_sequences(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _repair_double_triplet_anchor_sequences(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _regularize_repeated_note_rhythm(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _refine_isolated_long_note_values(
        quantized_melody_notes,
        bpm_estimate=rhythm.bpm_estimate,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _regularize_repeated_phrase_windows(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    quantized_melody_notes = _post_correct_mixed_subdivision_short_pitches(
        quantized_melody_notes,
        allowed_midis=allowed_midis,
    )
    quantized_melody_notes = _repair_same_pitch_dotted_tail_sequences(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    quantized_melody_notes = _repair_post_dotted_rearticulation(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    dotted_context_enabled = _has_reliable_dotted_anchors(quantized_melody_notes)
    if not dotted_context_enabled:
        quantized_melody_notes = _suppress_unreliable_dotted_patterns(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
        dotted_context_enabled = _has_reliable_dotted_anchors(quantized_melody_notes)
    if dotted_context_enabled:
        quantized_melody_notes = _repair_adjacent_isolated_dotted_pairs(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _repair_contextual_dotted_pairs(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _repair_same_pitch_dotted_tail_sequences(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
        quantized_melody_notes = _repair_post_dotted_rearticulation(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    quantized_melody_notes = _recover_missing_notes_from_raw_clusters(
        quantized_melody_notes,
        raw_melody_notes,
        bpm_estimate=rhythm.bpm_estimate,
        beats_per_measure=rhythm.beats_per_measure,
        grid_origin_seconds=rhythm.grid_origin_seconds,
    )
    quantized_melody_notes = _resolve_non_triplet_subbeat_patterns(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
        allow_dotted=False,
    )
    quantized_melody_notes = _resolve_monophonic_mixed_subdivision_collisions(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _repair_same_pitch_dotted_tail_sequences(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    quantized_melody_notes = _repair_post_dotted_rearticulation(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    if not _has_reliable_dotted_anchors(quantized_melody_notes):
        quantized_melody_notes = _suppress_unreliable_dotted_patterns(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    if _has_reliable_dotted_anchors(quantized_melody_notes):
        quantized_melody_notes = _repair_adjacent_isolated_dotted_pairs(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _repair_contextual_dotted_pairs(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _repair_same_pitch_dotted_tail_sequences(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
        quantized_melody_notes = _repair_post_dotted_rearticulation(
            quantized_melody_notes,
            beats_per_measure=rhythm.beats_per_measure,
        )
        quantized_melody_notes = _annotate_rhythm_groups(quantized_melody_notes)
    quantized_melody_notes = _repair_illegal_barline_splits(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _normalize_measure_grid(
        quantized_melody_notes,
        beats_per_measure=rhythm.beats_per_measure,
    )
    quantized_melody_notes = _annotate_articulation_candidates(
        quantized_melody_notes,
        frame_pitch_debug=frame_pitch_debug,
        audio=audio,
        sample_rate=preprocess.sample_rate,
        source_classification=source_classification,
    )

    sanity_warnings = _sanity_check_notes(quantized_melody_notes)
    notation_analysis = _build_notation_analysis(quantized_melody_notes, rhythm)
    core_inference = _core_inference_components()
    heuristic_corrections = _heuristic_correction_components()
    validation_observations = _build_validation_observations(candidate_note_names)

    ukulele_mapping: list[UkuleleMappedNote] = []
    tab_preview = "Tab preview skipped until segmented notes are manually reviewed."
    print("ukulele mapping skipped until segmented notes are reviewed.")

    intermediate_files = write_intermediate_outputs(
        output_dir=out_dir,
        frame_pitch_debug=frame_pitch_debug,
        pitch_histogram=pitch_histogram,
        source_classification=source_classification,
        rhythm=rhythm,
        raw_notes=raw_melody_notes,
        quantized_notes=quantized_melody_notes,
        notation_analysis=notation_analysis,
        core_inference=core_inference,
        heuristic_corrections=heuristic_corrections,
        validation_observations=validation_observations,
        segmentation_stats=segmentation_stats,
        tab_preview=tab_preview,
    )

    pdf_output_path: str | None = None
    pdf_ready = False
    pdf_skipped_reason: str | None = None
    try:
        pdf_output_path = _render_draft_tab_pdf(
            out_dir / f"{path.stem}_draft_tab.pdf",
            source_title=path.stem,
            notes=quantized_melody_notes,
            rhythm=rhythm,
        )
        pdf_ready = True
    except RuntimeError as exc:
        pdf_skipped_reason = str(exc)

    result = MelodyPipelineResult(
        output_type="intermediate_analysis_only",
        schema_name="melody_pipeline_result",
        source_path=str(path),
        output_dir=str(out_dir),
        preprocess=preprocess,
        source_classification=source_classification,
        rhythm_analysis=rhythm,
        melody_notes=quantized_melody_notes,
        raw_melody_notes=raw_melody_notes,
          ukulele_mapping=ukulele_mapping,
          tab_preview=tab_preview,
          analysis_notes=[
              "This stage prioritizes intermediate analysis files over final PDF rendering.",
              "Raw segmented notes and quantized notes are now written separately.",
              "Segmentation applies onset-aware splitting, same-pitch merging, and short-fragment filtering.",
              "Notation analysis now separates page layout, symbol events, and rhythmic values.",
              "The output now separates core inference from heuristic correction so overfitting risk is visible.",
              f"Source classification guess: {source_classification.source_family} / {source_classification.texture_type}.",
              f"Recommended pipeline: {source_classification.recommended_pipeline}.",
              "Ukulele mapping is intentionally disabled until segmented notes are reviewed.",
              "PDF export writes the current quantized note result as an auto-generated draft.",
              *(
                  [f"Closed-set note candidates active: {', '.join(candidate_note_names)}."]
                  if candidate_note_names
                else []
              ),
          ],
          core_inference=core_inference,
          heuristic_corrections=heuristic_corrections,
          validation_observations=validation_observations,
          sanity_warnings=sanity_warnings,
          pdf_ready=pdf_ready,
        pdf_skipped_reason=pdf_skipped_reason,
        notation_analysis=notation_analysis,
        pdf_output_path=pdf_output_path,
    )
    result_dict = result.to_dict()
    result_dict["intermediate_files"] = intermediate_files
    return result_dict
