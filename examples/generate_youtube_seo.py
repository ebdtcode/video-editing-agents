#!/usr/bin/env python3
"""
Example script demonstrating YouTube SEO metadata generation
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import ProcessingConfig, YouTubeSEOConfig
from src.agents.youtube_seo_agent import YouTubeSEOAgent


def main():
    """Generate YouTube SEO metadata from cleaned transcript"""

    # Configure YouTube SEO settings
    config = ProcessingConfig()

    # Customize SEO settings
    config.youtube_seo = YouTubeSEOConfig(
        enabled=True,
        title_max_length=100,
        description_min_length=200,
        description_max_length=5000,
        max_tags=30,
        include_keywords=True,
        include_timestamps=True,
        tone="engaging"  # Options: 'professional', 'engaging', 'casual'
    )

    # Initialize agent
    seo_agent = YouTubeSEOAgent(config)

    # Path to cleaned transcript
    transcript_path = "temp_segments/cleaned_transcript.txt"

    if not Path(transcript_path).exists():
        print(f"Error: Transcript file not found at {transcript_path}")
        print("Please run the video processing pipeline first to generate a cleaned transcript.")
        return

    print("🎬 Generating YouTube SEO metadata...")
    print(f"📄 Reading transcript from: {transcript_path}\n")

    # Generate metadata
    metadata = seo_agent.generate_metadata(
        transcript_path=transcript_path,
        video_duration=253.68  # Optional: provide video duration in seconds
    )

    # Display results
    print("=" * 80)
    print("📺 YOUTUBE SEO METADATA")
    print("=" * 80)
    print()

    print("🏷️  TITLE:")
    print("-" * 80)
    print(metadata.title)
    print()

    print("📝 DESCRIPTION:")
    print("-" * 80)
    print(metadata.description)
    print()

    print("🏷️  TAGS:")
    print("-" * 80)
    print(", ".join(metadata.tags))
    print()

    print("🔑 KEYWORDS:")
    print("-" * 80)
    print(", ".join(metadata.keywords))
    print()

    # Save to file
    output_path = "temp_segments/youtube_metadata.txt"
    seo_agent.save_metadata(metadata, output_path)

    print("=" * 80)
    print(f"✅ Metadata saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
