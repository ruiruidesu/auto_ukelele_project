from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.transcriber import (
    build_chord_sheet_schema,
    build_melody_pipeline,
    read_audio_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ukulele Auto Tab prototype")
    parser.add_argument("input_path", help="Path to a local audio file")
    parser.add_argument(
        "--mode",
        choices=["chord_sheet", "fingerstyle_tab"],
        default="fingerstyle_tab",
        help="Choose output mode: chord_sheet or fingerstyle_tab",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory for generated intermediate analysis files",
    )
    parser.add_argument(
        "--candidate-notes",
        help="Optional comma-separated note names, e.g. B4,A4,D5,E4,G4. When provided, melody classification is constrained to this note set.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input_path)

    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        return 1

    print(f"Processing file: {input_path}")
    print(f"Selected mode: {args.mode}")
    audio_path = read_audio_file(input_path)

    if args.mode == "chord_sheet":
        result = build_chord_sheet_schema(audio_path)
    else:
        candidate_notes = None
        if args.candidate_notes:
            candidate_notes = [item.strip() for item in args.candidate_notes.split(",") if item.strip()]
        result = build_melody_pipeline(
            audio_path,
            output_dir=args.output_dir,
            candidate_note_names=candidate_notes,
        )

    print(f"Output type: {result['output_type']}")
    print(f"Schema: {result['schema_name']}")
    if "analysis_notes" in result:
        print("Analysis notes:")
        for note in result["analysis_notes"]:
            print(f"  - {note}")
    if "source_classification" in result:
        source_info = result["source_classification"]
        print("Source classification:")
        print(
            "  "
            f"{source_info['source_family']} / {source_info['texture_type']} "
            f"(recommended: {source_info['recommended_pipeline']})"
        )
    if "text_tab_preview" in result:
        print("Text tab preview:")
        print(result["text_tab_preview"])
    if "tab_preview" in result:
        print("Tab preview:")
        print(result["tab_preview"])
    if "intermediate_files" in result:
        print("Intermediate files:")
        for key, value in result["intermediate_files"].items():
            print(f"  {key}: {value}")
    if result.get("pdf_ready"):
        print("PDF status: ready")
        print(f"PDF path: {result.get('pdf_output_path')}")
    elif "pdf_skipped_reason" in result:
        print(f"PDF status: skipped")
        print(f"Reason: {result['pdf_skipped_reason']}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
