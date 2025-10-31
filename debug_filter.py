"""Debug script to print the exact filter_complex being generated"""

# Simulate 3 segments with known durations
segments = [
    {"tts_duration": 14.20},
    {"tts_duration": 14.04},
    {"tts_duration": 7.56},
]

duration = 0.3  # transition duration
xfade_transition = "fade"

# Build labels
video_labels = [f"[{i}:v]" for i in range(len(segments))]
audio_labels = [f"[{i}:a]" for i in range(len(segments))]

video_filter_parts = []
audio_filter_parts = []

# Video xfade (same as before)
current_video_label = video_labels[0]
cumulative_offset = 0

for i in range(len(segments) - 1):
    next_video_label = video_labels[i + 1]
    output_video_label = f"[v{i}]" if i < len(segments) - 2 else "[outv]"
    
    offset = cumulative_offset + segments[i]["tts_duration"] - duration
    cumulative_offset = offset + duration
    
    video_filter_parts.append(
        f"{current_video_label}{next_video_label}xfade="
        f"transition={xfade_transition}:"
        f"duration={duration}:"
        f"offset={offset:.2f}"
        f"{output_video_label}"
    )
    
    current_video_label = output_video_label

# Audio - current implementation
trimmed_audio_labels = []

for i, segment in enumerate(segments):
    trimmed_label = f"[atrim{i}]"
    
    if i < len(segments) - 1:
        trim_end = segment["tts_duration"] - duration
        audio_filter_parts.append(
            f"{audio_labels[i]}atrim=0:{trim_end},"
            f"asetpts=PTS-STARTPTS{trimmed_label}"
        )
    else:
        audio_filter_parts.append(
            f"{audio_labels[i]}asetpts=PTS-STARTPTS{trimmed_label}"
        )
    
    trimmed_audio_labels.append(trimmed_label)

audio_inputs = ''.join(trimmed_audio_labels)
audio_filter_parts.append(
    f"{audio_inputs}concat=n={len(segments)}:v=0:a=1[outa]"
)

# Print results
print("=" * 80)
print("VIDEO FILTERS:")
print("=" * 80)
for f in video_filter_parts:
    print(f)

print("\n" + "=" * 80)
print("AUDIO FILTERS:")
print("=" * 80)
for f in audio_filter_parts:
    print(f)

print("\n" + "=" * 80)
print("FULL FILTER_COMPLEX:")
print("=" * 80)
all_filters = video_filter_parts + audio_filter_parts
filter_complex = ';'.join(all_filters)
print(filter_complex)

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)
print(f"Segment 0: {segments[0]['tts_duration']}s -> trim to {segments[0]['tts_duration'] - duration}s")
print(f"Segment 1: {segments[1]['tts_duration']}s -> trim to {segments[1]['tts_duration'] - duration}s")
print(f"Segment 2: {segments[2]['tts_duration']}s -> no trim (last segment)")
print(f"\nExpected audio duration: {(segments[0]['tts_duration'] - duration) + (segments[1]['tts_duration'] - duration) + segments[2]['tts_duration']}s")
print(f"Expected video duration: {sum(s['tts_duration'] for s in segments) - (len(segments) - 1) * duration}s")
