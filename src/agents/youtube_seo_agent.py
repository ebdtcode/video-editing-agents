"""
YouTube SEO Agent - Generates SEO-optimized metadata for YouTube videos
"""

from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import re

from src.config import YouTubeSEOConfig, ProcessingConfig
from src.exceptions import ValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class YouTubeMetadata:
    """Represents YouTube SEO metadata"""
    title: str
    description: str
    tags: List[str]
    keywords: List[str]


class YouTubeSEOAgent:
    """Generates SEO-optimized YouTube metadata from video transcripts"""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize YouTube SEO agent

        Args:
            config: Processing configuration
        """
        self.config = config.youtube_seo
        self.logger = get_logger(self.__class__.__name__)

    def generate_metadata(
        self,
        transcript_path: str,
        video_duration: Optional[float] = None
    ) -> YouTubeMetadata:
        """
        Generate YouTube SEO metadata from transcript

        Args:
            transcript_path: Path to cleaned transcript file
            video_duration: Optional video duration in seconds

        Returns:
            YouTubeMetadata object with title, description, tags, and keywords

        Raises:
            ValidationError: If transcript is invalid or empty
        """
        if not self.config.enabled:
            self.logger.info("YouTube SEO generation is disabled")
            return YouTubeMetadata(title="", description="", tags=[], keywords=[])

        # Read transcript
        transcript = self._read_transcript(transcript_path)

        if not transcript:
            raise ValidationError("Transcript is empty")

        self.logger.info(f"Generating YouTube SEO metadata from transcript ({len(transcript)} lines)")

        # Extract content
        content = self._extract_content(transcript)

        # Generate components
        title = self._generate_title(content)
        keywords = self._extract_keywords(content)
        tags = self._generate_tags(content, keywords)
        description = self._generate_description(content, keywords, transcript, video_duration)

        metadata = YouTubeMetadata(
            title=title,
            description=description,
            tags=tags,
            keywords=keywords
        )

        self.logger.info(f"Generated metadata: title={title[:50]}..., tags={len(tags)}, keywords={len(keywords)}")

        return metadata

    def _read_transcript(self, transcript_path: str) -> List[str]:
        """Read transcript file and return lines"""
        path = Path(transcript_path)

        if not path.exists():
            raise ValidationError(f"Transcript file not found: {transcript_path}")

        with open(path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        return lines

    def _extract_content(self, transcript: List[str]) -> str:
        """Extract plain text content from transcript"""
        text_parts = []

        for line in transcript:
            # Remove timestamp patterns like [1.25s - 17.43s]
            text = re.sub(r'\[[\d.s\- ]+\]', '', line)
            text = text.strip()
            if text:
                text_parts.append(text)

        return ' '.join(text_parts)

    def _generate_title(self, content: str) -> str:
        """
        Generate SEO-friendly YouTube title

        Strategy:
        1. Extract key topic from first sentence or most important phrase
        2. Make it engaging and clickable
        3. Keep under max_length limit
        """
        # Extract first meaningful sentence
        sentences = re.split(r'[.!?]+', content)
        first_sentence = sentences[0].strip() if sentences else content[:100]

        # Remove common filler phrases
        first_sentence = re.sub(r'^(hello|hi|hey|welcome|today)\s+(everyone|guys|folks)?,?\s*', '',
                               first_sentence, flags=re.IGNORECASE)
        first_sentence = re.sub(r'^(welcome\s+to\s+the\s+channel|welcome\s+back)\s*,?\s*', '',
                               first_sentence, flags=re.IGNORECASE)

        # Capitalize first letter
        if first_sentence:
            first_sentence = first_sentence[0].upper() + first_sentence[1:]

        # Identify main topic
        title = self._extract_main_topic(first_sentence, content)

        # Ensure title is within length limit
        if len(title) > self.config.title_max_length:
            title = title[:self.config.title_max_length - 3] + "..."

        return title

    def _extract_main_topic(self, first_sentence: str, full_content: str) -> str:
        """Extract the main topic from content"""
        # Look for patterns like "reviewing X", "about X", "how to X"
        patterns = [
            r'(?:review|reviewing|about|discussing)\s+(?:the\s+)?([^,\.]+)',
            r'how\s+to\s+([^,\.]+)',
            r'(?:today|now)\s+(?:we\'re|we are|i\'m|i am)\s+([^,\.]+)',
        ]

        for pattern in patterns:
            match = re.search(pattern, first_sentence, re.IGNORECASE)
            if match:
                topic = match.group(1).strip()
                return self._format_title(topic)

        # Fallback: use first sentence
        return first_sentence[:self.config.title_max_length]

    def _format_title(self, topic: str) -> str:
        """Format topic into engaging title"""
        # Capitalize important words
        words = topic.split()
        formatted_words = []

        skip_words = {'a', 'an', 'the', 'and', 'or', 'but', 'for', 'with', 'to', 'from', 'in', 'on', 'at'}

        for i, word in enumerate(words):
            if i == 0 or word.lower() not in skip_words:
                formatted_words.append(word.capitalize())
            else:
                formatted_words.append(word.lower())

        return ' '.join(formatted_words)

    def _extract_keywords(self, content: str) -> List[str]:
        """Extract important keywords from content"""
        if not self.config.include_keywords:
            return []

        # Convert to lowercase for analysis
        content_lower = content.lower()

        # Extract potential keywords (nouns and important phrases)
        words = re.findall(r'\b[a-z]{3,}\b', content_lower)

        # Count word frequency
        word_freq = {}
        for word in words:
            if word not in self._get_stop_words():
                word_freq[word] = word_freq.get(word, 0) + 1

        # Sort by frequency and take top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        keywords = [word for word, _ in sorted_keywords[:15]]

        return keywords

    def _generate_tags(self, content: str, keywords: List[str]) -> List[str]:
        """Generate YouTube tags from content and keywords"""
        tags = []

        # Add top keywords as single-word tags
        for keyword in keywords[:10]:
            if keyword not in tags:
                tags.append(keyword)

        # Extract multi-word phrases
        phrases = self._extract_phrases(content)
        for phrase in phrases:
            if len(tags) < self.config.max_tags:
                tags.append(phrase)

        # Limit to max_tags
        return tags[:self.config.max_tags]

    def _extract_phrases(self, content: str) -> List[str]:
        """Extract common multi-word phrases"""
        # Find 2-3 word combinations that appear multiple times
        words = content.lower().split()
        phrases = {}

        # Extract 2-word phrases
        for i in range(len(words) - 1):
            phrase = f"{words[i]} {words[i+1]}"
            if len(phrase) > 6 and not any(sw in phrase.split() for sw in self._get_stop_words()):
                phrases[phrase] = phrases.get(phrase, 0) + 1

        # Extract 3-word phrases
        for i in range(len(words) - 2):
            phrase = f"{words[i]} {words[i+1]} {words[i+2]}"
            if len(phrase) > 10:
                phrases[phrase] = phrases.get(phrase, 0) + 1

        # Sort by frequency and return top phrases
        sorted_phrases = sorted(phrases.items(), key=lambda x: x[1], reverse=True)
        return [phrase for phrase, count in sorted_phrases if count >= 2][:10]

    def _generate_description(
        self,
        content: str,
        keywords: List[str],
        transcript: List[str],
        video_duration: Optional[float] = None
    ) -> str:
        """Generate engaging YouTube description"""
        description_parts = []

        # 1. Opening hook (first 2-3 sentences summarized)
        sentences = re.split(r'[.!?]+', content)
        opening = self._create_opening_hook(sentences[:3])
        description_parts.append(opening)
        description_parts.append("")

        # 2. What viewers will learn/see
        description_parts.append(self._create_value_proposition(content, keywords))
        description_parts.append("")

        # 3. Timestamps (if enabled)
        if self.config.include_timestamps and video_duration:
            timestamps = self._create_timestamps(transcript, video_duration)
            if timestamps:
                description_parts.append("⏱️ TIMESTAMPS:")
                description_parts.extend(timestamps)
                description_parts.append("")

        # 4. Call to action
        description_parts.append(self._create_call_to_action())
        description_parts.append("")

        # 5. Keywords section
        if self.config.include_keywords and keywords:
            description_parts.append("📌 TAGS:")
            description_parts.append(", ".join(keywords))

        description = "\n".join(description_parts)

        # Ensure description meets length requirements
        if len(description) < self.config.description_min_length:
            description += "\n\n" + self._add_padding_content(content)

        if len(description) > self.config.description_max_length:
            description = description[:self.config.description_max_length - 3] + "..."

        return description

    def _create_opening_hook(self, first_sentences: List[str]) -> str:
        """Create engaging opening for description"""
        # Clean up sentences
        cleaned = []
        for sentence in first_sentences:
            sentence = sentence.strip()
            if sentence:
                # Remove common intros
                sentence = re.sub(r'^(hello|hi|hey|welcome)\s+(everyone|guys|folks)?,?\s*', '',
                                 sentence, flags=re.IGNORECASE)
                if sentence:
                    cleaned.append(sentence)

        if not cleaned:
            return "Watch this video to learn more!"

        hook = '. '.join(cleaned[:2]) + '.'
        return hook

    def _create_value_proposition(self, content: str, keywords: List[str]) -> str:
        """Create 'What you'll learn' section"""
        if not keywords:
            return "In this video, you'll discover valuable insights and information."

        value_phrases = [
            "In this video, you'll discover:",
            "✨ What you'll learn:",
            "🎯 Key topics covered:",
        ]

        tone_index = 0 if self.config.tone == "professional" else (1 if self.config.tone == "engaging" else 2)
        intro = value_phrases[min(tone_index, len(value_phrases) - 1)]

        # Create bullet points from keywords
        bullets = [f"• {keyword.capitalize()}" for keyword in keywords[:5]]

        return intro + "\n" + "\n".join(bullets)

    def _create_timestamps(self, transcript: List[str], video_duration: float) -> List[str]:
        """Create timestamp sections from transcript"""
        timestamps = []

        for i, line in enumerate(transcript[:10]):  # First 10 segments
            # Extract timestamp from line like [1.25s - 17.43s]
            match = re.search(r'\[([\d.]+)s\s*-\s*([\d.]+)s\]', line)
            if match:
                start_time = float(match.group(1))
                # Extract text
                text = re.sub(r'\[[\d.s\- ]+\]', '', line).strip()

                # Format timestamp
                minutes = int(start_time // 60)
                seconds = int(start_time % 60)
                timestamp = f"{minutes:02d}:{seconds:02d}"

                # Create timestamp entry (first 50 chars of text)
                if text:
                    timestamps.append(f"{timestamp} - {text[:50]}")

        return timestamps

    def _create_call_to_action(self) -> str:
        """Create call to action based on tone"""
        ctas = {
            "professional": "If you found this helpful, please subscribe for more content.",
            "engaging": "👍 Don't forget to LIKE and SUBSCRIBE for more amazing content!",
            "casual": "Hope you enjoyed! Drop a like and subscribe if you want more!"
        }

        return ctas.get(self.config.tone, ctas["engaging"])

    def _add_padding_content(self, content: str) -> str:
        """Add additional content to meet minimum length"""
        # Extract a summary paragraph
        sentences = re.split(r'[.!?]+', content)
        summary = '. '.join(sentences[:5]) + '.'

        return f"📝 MORE DETAILS:\n{summary}"

    def _get_stop_words(self) -> set:
        """Return common English stop words"""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were', 'been',
            'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'should', 'could', 'may', 'might', 'must', 'can', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'when', 'where', 'why', 'how', 'all', 'each',
            'every', 'both', 'few', 'more', 'most', 'other', 'some', 'such',
            'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
            'very', 'just', 'now', 'here', 'there', 'then', 'also'
        }

    def save_metadata(self, metadata: YouTubeMetadata, output_path: str):
        """
        Save metadata to a file

        Args:
            metadata: YouTubeMetadata object
            output_path: Path to save metadata
        """
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', encoding='utf-8') as f:
            f.write(f"TITLE:\n{metadata.title}\n\n")
            f.write(f"DESCRIPTION:\n{metadata.description}\n\n")
            f.write(f"TAGS:\n{', '.join(metadata.tags)}\n\n")
            f.write(f"KEYWORDS:\n{', '.join(metadata.keywords)}\n")

        self.logger.info(f"Metadata saved to {output_path}")
