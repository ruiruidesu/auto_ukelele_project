from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class ChordAnchor:
    beat_offset: float
    lyric_char_index: int
    chord_name: str


@dataclass
class LyricLine:
    text: str
    chord_anchors: list[ChordAnchor]


@dataclass
class ChordSection:
    name: str
    lines: list[LyricLine]


@dataclass
class ChordSheetSchema:
    output_type: str
    schema_name: str
    source_path: str
    notes: str
    sections: list[ChordSection]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class AudioPreprocessSummary:
    source_path: str
    sample_rate: int
    channels: int
    trimmed_start_seconds: float
    trimmed_end_seconds: float
    duration_seconds: float
    normalized_peak: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class SourceClassification:
    source_family: str
    source_confidence: float
    texture_type: str
    texture_confidence: float
    vocal_presence: bool
    recommended_pipeline: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MelodyNote:
    note_index: int
    start_time: float
    end_time: float
    duration: float
    frequency_hz: float
    midi: int
    note_name: str
    confidence: float
    raw_beat_position: float
    quantized_beat_position: float
    quantized_duration_beats: float
    measure_number: int
    beat_in_measure: float
    onset_supported: bool = False
    offset_supported: bool = False
    onset_strength: float = 0.0
    offset_strength: float = 0.0
    observed_frequency_hz: float = 0.0
    rhythm_value: str = ""
    beam_group: str | None = None
    beat_start_in_measure: float = 0.0
    duration_beats: float = 0.0
    string_name: str | None = None
    fret_number: int | None = None
    beat_pattern: str | None = None
    tuplet_group: str | None = None
    slide_to_next: bool = False
    slide_from_previous: bool = False
    harmonic_candidate: bool = False
    articulation_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TabSystemLayout:
    system_index: int
    measure_numbers: list[int]
    string_count: int
    string_labels: list[str]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TabPageLayout:
    system_count: int
    systems: list[TabSystemLayout]
    beats_per_measure: int
    bpm_estimate: float
    time_signature: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TabSymbolEvent:
    note_index: int
    measure_number: int
    string_name: str
    fret_number: int
    beat_start_in_measure: float
    duration_beats: float
    rhythm_value: str
    beam_group: str | None
    beam_level: int
    beat_pattern: str | None = None
    tuplet_group: str | None = None
    slide_to_next: bool = False
    slide_from_previous: bool = False
    harmonic_candidate: bool = False
    articulation_hint: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class TabNotationAnalysis:
    page_layout: TabPageLayout
    symbol_events: list[TabSymbolEvent]
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class RhythmAnalysis:
    bpm_estimate: float
    bpm_confidence_note: str
    beat_positions: list[float]
    strong_beat_positions: list[float]
    measure_boundaries: list[float]
    beats_per_measure: int
    grid_origin_seconds: float
    quantization_grid: str
    time_signature_tendency: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class UkuleleMappedNote:
    note_index: int
    note_name: str
    midi: int
    start_time: float
    duration: float
    measure_number: int
    beat_in_measure: float
    quantized_duration_beats: float
    chosen_string: str
    chosen_string_number: int
    chosen_fret: int
    hand_position_fret: int
    mapping_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class InferenceComponent:
    name: str
    category: str
    description: str
    generalization_note: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class ValidationObservation:
    sample_name: str
    validation_type: str
    status: str
    notes: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class MelodyPipelineResult:
    output_type: str
    schema_name: str
    source_path: str
    output_dir: str
    preprocess: AudioPreprocessSummary
    source_classification: SourceClassification
    rhythm_analysis: RhythmAnalysis
    melody_notes: list[MelodyNote]
    raw_melody_notes: list[MelodyNote]
    ukulele_mapping: list[UkuleleMappedNote]
    tab_preview: str
    analysis_notes: list[str]
    core_inference: list[InferenceComponent]
    heuristic_corrections: list[InferenceComponent]
    validation_observations: list[ValidationObservation]
    sanity_warnings: list[str]
    pdf_ready: bool
    pdf_skipped_reason: str | None
    notation_analysis: TabNotationAnalysis | None = None
    pdf_output_path: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)
