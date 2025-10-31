#!/usr/bin/env python3
"""Analyze TTS failures by examining failed segment text."""

import json
from pathlib import Path

# Failed segment IDs from the test report
FAILED_SEGMENTS = [
    "seg_0000", "seg_0001", "seg_0005", "seg_0007", "seg_0008",
    "seg_0009", "seg_0010", "seg_0011", "seg_0012", "seg_0013",
    "seg_0016", "seg_0017", "seg_0021", "seg_0024", "seg_0028",
    "seg_0029", "seg_0030", "seg_0034"
]

def analyze_failures():
    """Analyze text from failed TTS segments."""

    # Read cleaned transcript
    transcript_path = Path("temp_segments/cleaned_transcript.txt")
    if not transcript_path.exists():
        print(f"❌ Transcript not found: {transcript_path}")
        return

    with open(transcript_path, 'r', encoding='utf-8') as f:
        content = f.read()

    print("=" * 80)
    print("FAILED SEGMENT ANALYSIS")
    print("=" * 80)
    print(f"\nTotal failed segments: {len(FAILED_SEGMENTS)}")
    print(f"Failure rate: {len(FAILED_SEGMENTS)}/39 = {len(FAILED_SEGMENTS)/39*100:.1f}%\n")

    # Parse segments from transcript
    # Format: [start_time - end_time] text
    segments = {}
    import re

    # Split by lines and parse
    lines = content.strip().split('\n')
    for i, line in enumerate(lines):
        if not line.strip():
            continue

        # Extract text after timestamp
        match = re.search(r'\[([\d.]+)s - ([\d.]+)s\] (.+)', line)
        if match:
            seg_id = f"seg_{i:04d}"
            text = match.group(3).strip()
            segments[seg_id] = text

    print(f"Total segments in transcript: {len(segments)}\n")

    # Analyze failed segments
    print("FAILED SEGMENT DETAILS:")
    print("-" * 80)

    text_lengths = []
    empty_count = 0
    short_count = 0  # < 3 chars
    special_char_count = 0

    for seg_id in FAILED_SEGMENTS:
        text = segments.get(seg_id, "[NOT FOUND]")
        text_len = len(text)
        word_count = len(text.split())

        text_lengths.append(text_len)

        # Categorize
        is_empty = text_len == 0 or text == "[NOT FOUND]"
        is_short = text_len < 3 and text_len > 0
        has_special = any(c in text for c in ['[', ']', '{', '}', '<', '>', '|'])

        if is_empty:
            empty_count += 1
            status = "❌ EMPTY"
        elif is_short:
            short_count += 1
            status = "⚠️  SHORT"
        elif has_special:
            special_char_count += 1
            status = "⚠️  SPECIAL_CHARS"
        else:
            status = "❓ UNKNOWN"

        print(f"\n{seg_id}: {status}")
        print(f"  Length: {text_len} chars, {word_count} words")
        print(f"  Text: '{text[:100]}'")
        if text_len > 100:
            print(f"        ... (truncated)")

    # Summary statistics
    print("\n" + "=" * 80)
    print("FAILURE PATTERN SUMMARY:")
    print("-" * 80)
    print(f"Empty/Not Found:  {empty_count:2d} ({empty_count/len(FAILED_SEGMENTS)*100:.1f}%)")
    print(f"Too Short (<3ch): {short_count:2d} ({short_count/len(FAILED_SEGMENTS)*100:.1f}%)")
    print(f"Special Chars:    {special_char_count:2d} ({special_char_count/len(FAILED_SEGMENTS)*100:.1f}%)")
    print(f"Unknown Cause:    {len(FAILED_SEGMENTS)-empty_count-short_count:2d}")

    if text_lengths:
        print(f"\nText Length Stats:")
        print(f"  Min:  {min(text_lengths)} chars")
        print(f"  Max:  {max(text_lengths)} chars")
        print(f"  Avg:  {sum(text_lengths)/len(text_lengths):.1f} chars")

    # Compare with successful segments
    print("\n" + "=" * 80)
    print("SUCCESSFUL SEGMENT COMPARISON:")
    print("-" * 80)

    successful_segments = [seg_id for seg_id in segments.keys() if seg_id not in FAILED_SEGMENTS]
    successful_texts = [segments[seg_id] for seg_id in successful_segments[:5]]

    print(f"Total successful segments: {len(successful_segments)}\n")
    print("Sample successful segment texts:")
    for i, text in enumerate(successful_texts, 1):
        print(f"\n{i}. Length: {len(text)} chars")
        print(f"   Text: '{text[:100]}'")
        if len(text) > 100:
            print(f"         ... (truncated)")

if __name__ == "__main__":
    analyze_failures()
