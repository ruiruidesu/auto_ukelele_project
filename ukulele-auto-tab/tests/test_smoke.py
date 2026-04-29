from __future__ import annotations

import json
import shutil
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np


def _make_test_dir(test_name: str) -> Path:
    project_root = Path(__file__).resolve().parents[1]
    test_dir = project_root / "tests_runtime" / test_name
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True, exist_ok=True)
    return test_dir


def _write_test_tone(audio_file: Path, frequencies: list[float], sample_rate: int = 22050) -> None:
    segment_duration = 0.45
    silence_duration = 0.04
    samples: list[np.ndarray] = []
    for frequency in frequencies:
        tone_time = np.linspace(0, segment_duration, int(sample_rate * segment_duration), endpoint=False)
        silence = np.zeros(int(sample_rate * silence_duration), dtype=np.float32)
        waveform = 0.4 * np.sin(2 * np.pi * frequency * tone_time)
        samples.append(waveform.astype(np.float32))
        samples.append(silence)
    full_wave = np.concatenate(samples) if samples else np.zeros(sample_rate, dtype=np.float32)
    pcm_audio = np.int16(full_wave * 32767)

    with wave.open(str(audio_file), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


def test_main_with_missing_file_returns_error() -> None:
    project_root = Path(__file__).resolve().parents[1]
    missing_file = project_root / "data" / "missing-demo-file.wav"

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "--mode",
            "fingerstyle_tab",
            str(missing_file),
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 1
    assert f"Error: input file not found: {missing_file}" in result.stdout


def test_chord_sheet_mode_stays_placeholder() -> None:
    project_root = Path(__file__).resolve().parents[1]
    test_dir = _make_test_dir("chord_sheet_mode")
    audio_file = test_dir / "placeholder.wav"
    _write_test_tone(audio_file, [440.0, 493.88])

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "--mode",
            "chord_sheet",
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    assert result.returncode == 0
    assert '"schema_name": "chord_sheet_schema"' in result.stdout
    assert '"output_type": "placeholder_sheet_structure"' in result.stdout


def test_fingerstyle_mode_writes_intermediate_files() -> None:
    project_root = Path(__file__).resolve().parents[1]
    test_dir = _make_test_dir("fingerstyle_mode")
    audio_file = test_dir / "melody.wav"
    output_dir = test_dir / "generated"
    _write_test_tone(audio_file, [440.0, 493.88, 523.25, 587.33])

    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "main.py"),
            "--mode",
            "fingerstyle_tab",
            "--output-dir",
            str(output_dir),
            str(audio_file),
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
    )

    melody_debug = output_dir / "melody_notes_debug.txt"
    pitch_histogram = output_dir / "pitch_histogram.txt"
    segmented_notes_raw = output_dir / "segmented_notes_raw.json"
    segmented_notes_quantized = output_dir / "segmented_notes_quantized.json"
    segmentation_report = output_dir / "segmentation_report.txt"
    mapping_debug = output_dir / "ukulele_mapping_debug.txt"
    tab_txt = output_dir / "tab_preview.txt"
    draft_pdf = output_dir / f"{audio_file.stem}_draft_tab.pdf"

    assert result.returncode == 0
    assert "Output type: intermediate_analysis_only" in result.stdout
    assert ("PDF status: skipped" in result.stdout) or ("PDF status: ready" in result.stdout)
    assert melody_debug.exists()
    assert pitch_histogram.exists()
    assert segmented_notes_raw.exists()
    assert segmented_notes_quantized.exists()
    assert segmentation_report.exists()
    assert mapping_debug.exists()
    assert tab_txt.exists()
    if "PDF status: ready" in result.stdout:
        assert draft_pdf.exists()

    raw_melody_data = json.loads(segmented_notes_raw.read_text(encoding="utf-8"))
    quantized_melody_data = json.loads(segmented_notes_quantized.read_text(encoding="utf-8"))
    mapping_debug_text = mapping_debug.read_text(encoding="utf-8")
    melody_debug_text = melody_debug.read_text(encoding="utf-8")
    pitch_histogram_text = pitch_histogram.read_text(encoding="utf-8")
    segmentation_report_text = segmentation_report.read_text(encoding="utf-8")
    tab_preview = tab_txt.read_text(encoding="utf-8")

    assert len(raw_melody_data) >= 3
    assert len(quantized_melody_data) >= 3
    assert raw_melody_data[0]["note_name"].startswith("A")
    assert all("start_time" in note for note in raw_melody_data)
    assert all("duration" in note for note in raw_melody_data)
    assert all("quantized_beat_position" in note for note in quantized_melody_data)
    assert "Frame-level pitch estimates" in melody_debug_text
    assert "Segmented note-level timeline" in melody_debug_text
    assert "Pitch histogram" in pitch_histogram_text
    assert "merged fragments:" in segmentation_report_text
    assert "filtered short notes:" in segmentation_report_text
    assert "final note count:" in segmentation_report_text
    assert "status: skipped_until_segmented_notes_are_reviewed" in mapping_debug_text
    assert "skipped until segmented notes are manually reviewed" in tab_preview.lower()
