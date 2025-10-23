"""
Content Analysis Agent - Analyzes and cleans transcripts
"""

from pathlib import Path
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass

from src.config import FillerConfig, ProcessingConfig
from src.agents.transcription_agent import Segment, Word
from src.exceptions import ValidationError
from src.utils.logger import get_logger
from src.utils.text_normalizer import TextNormalizer


logger = get_logger(__name__)


@dataclass
class CleanSegment:
    """Represents a cleaned segment with timing information"""
    segment_id: str
    text: str
    start: float
    end: float
    words: List[Word]
    original_text: str  # Text before cleaning


class ContentAnalysisAgent:
    """Analyzes transcripts and identifies content to remove"""

    def __init__(self, config: ProcessingConfig):
        """
        Initialize content analysis agent

        Args:
            config: Processing configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.text_normalizer = TextNormalizer()

    def detect_fillers(
        self,
        segments: List[Segment]
    ) -> Tuple[List[CleanSegment], Dict[str, int]]:
        """
        Detect and remove filler words from segments

        Args:
            segments: List of transcript segments

        Returns:
            Tuple of (clean segments, filler statistics)

        Raises:
            ValidationError: If segments are invalid
        """
        if not segments:
            raise ValidationError("No segments provided for filler detection")

        self.logger.info(f"Detecting fillers in {len(segments)} segments")

        # Flatten all words from all segments
        all_words = []
        for segment in segments:
            all_words.extend(segment.words)

        # Split into clean segments
        if self.config.fillers.mode == "rule-based":
            clean_segments, stats = self._detect_fillers_rule_based(all_words)
        else:
            # ML-based detection (placeholder for future implementation)
            self.logger.warning("ML-based filler detection not implemented, falling back to rule-based")
            clean_segments, stats = self._detect_fillers_rule_based(all_words)

        self.logger.info(
            f"Created {len(clean_segments)} clean segments. "
            f"Removed {stats['total_fillers']} filler words"
        )

        return clean_segments, stats

    def _detect_fillers_rule_based(
        self,
        words: List[Word]
    ) -> Tuple[List[CleanSegment], Dict[str, int]]:
        """
        Detect fillers using rule-based approach

        Args:
            words: List of all words

        Returns:
            Tuple of (clean segments, statistics)
        """
        filler_words = [f.lower() for f in self.config.fillers.words]
        filler_count = {word: 0 for word in filler_words}
        filler_count['other'] = 0

        clean_segments = []
        current_segment_words = []
        segment_counter = 0

        for word in words:
            word_lower = word.word.lower().strip()

            # Check if it's a filler word
            is_filler = word_lower in filler_words

            if is_filler:
                # Track which filler was found
                if word_lower in filler_count:
                    filler_count[word_lower] += 1
                else:
                    filler_count['other'] += 1

                # End current segment if we have words
                if current_segment_words:
                    segment = self._create_clean_segment(
                        segment_counter,
                        current_segment_words
                    )
                    if segment:
                        clean_segments.append(segment)
                        segment_counter += 1

                    current_segment_words = []
            else:
                # Add to current segment
                current_segment_words.append(word)

        # Add final segment
        if current_segment_words:
            segment = self._create_clean_segment(
                segment_counter,
                current_segment_words
            )
            if segment:
                clean_segments.append(segment)

        # Calculate statistics
        total_fillers = sum(filler_count.values())
        stats = {
            'total_fillers': total_fillers,
            'filler_breakdown': filler_count,
            'original_words': len(words),
            'clean_words': sum(len(s.words) for s in clean_segments),
            'removal_rate': total_fillers / len(words) if len(words) > 0 else 0
        }

        return clean_segments, stats

    def _create_clean_segment(
        self,
        segment_id: int,
        words: List[Word]
    ) -> Optional[CleanSegment]:
        """
        Create a clean segment from words

        Args:
            segment_id: Segment identifier
            words: List of words in segment

        Returns:
            CleanSegment or None if segment is too short
        """
        if not words:
            return None

        start = words[0].start
        end = words[-1].end
        duration = end - start

        # Skip very short segments
        if duration < self.config.min_segment_duration:
            self.logger.debug(
                f"Skipping segment {segment_id}: duration {duration:.2f}s "
                f"< minimum {self.config.min_segment_duration}s"
            )
            return None

        # Check maximum segment duration
        if duration > self.config.max_segment_duration:
            self.logger.debug(
                f"Segment {segment_id} exceeds maximum duration "
                f"({duration:.2f}s > {self.config.max_segment_duration}s), "
                "will be split"
            )
            # For now, we'll keep it as one segment
            # TODO: Implement smart splitting at sentence boundaries

        # Join words and normalize text for natural speech
        raw_text = " ".join(w.word for w in words)
        normalized_text = self.text_normalizer.normalize_segment_text(raw_text)

        return CleanSegment(
            segment_id=f"seg_{segment_id:04d}",
            text=normalized_text,
            start=start,
            end=end,
            words=words,
            original_text=raw_text  # Keep original for reference
        )

    def find_optimal_cuts(
        self,
        segments: List[CleanSegment]
    ) -> List[CleanSegment]:
        """
        Find optimal cut points for segments that are too long

        Args:
            segments: List of clean segments

        Returns:
            List of optimized segments
        """
        optimized = []

        for segment in segments:
            duration = segment.end - segment.start

            if duration <= self.config.max_segment_duration:
                optimized.append(segment)
            else:
                # Split long segment
                split_segments = self._split_segment(segment)
                optimized.extend(split_segments)

        self.logger.info(
            f"Optimized {len(segments)} segments into {len(optimized)} segments"
        )

        return optimized

    def _split_segment(
        self,
        segment: CleanSegment
    ) -> List[CleanSegment]:
        """
        Split a long segment into smaller segments

        Args:
            segment: Segment to split

        Returns:
            List of smaller segments
        """
        # Simple implementation: split at word boundaries
        target_duration = self.config.max_segment_duration
        split_segments = []

        current_words = []
        current_start = segment.start
        segment_counter = 0

        for word in segment.words:
            current_words.append(word)
            duration = word.end - current_start

            if duration >= target_duration and len(current_words) > 1:
                # Create segment with normalized text
                raw_text = " ".join(w.word for w in current_words)
                normalized_text = self.text_normalizer.normalize_segment_text(raw_text)

                new_segment = CleanSegment(
                    segment_id=f"{segment.segment_id}_split{segment_counter}",
                    text=normalized_text,
                    start=current_start,
                    end=current_words[-1].end,
                    words=current_words,
                    original_text=raw_text
                )
                split_segments.append(new_segment)

                # Reset for next segment
                current_words = []
                current_start = word.end
                segment_counter += 1

        # Add remaining words
        if current_words:
            raw_text = " ".join(w.word for w in current_words)
            normalized_text = self.text_normalizer.normalize_segment_text(raw_text)

            new_segment = CleanSegment(
                segment_id=f"{segment.segment_id}_split{segment_counter}",
                text=normalized_text,
                start=current_start,
                end=current_words[-1].end,
                words=current_words,
                original_text=raw_text
            )
            split_segments.append(new_segment)

        return split_segments

    def analyze_speech_patterns(
        self,
        segments: List[Segment]
    ) -> Dict[str, any]:
        """
        Analyze speech patterns in the transcript

        Args:
            segments: List of segments

        Returns:
            Dict with speech pattern analysis
        """
        if not segments:
            return {}

        total_words = sum(len(s.words) for s in segments)
        total_duration = sum(s.end - s.start for s in segments)

        # Calculate speaking rate
        words_per_minute = (total_words / total_duration * 60) if total_duration > 0 else 0

        # Calculate pauses (gaps between segments)
        pauses = []
        for i in range(len(segments) - 1):
            pause = segments[i + 1].start - segments[i].end
            if pause > 0:
                pauses.append(pause)

        avg_pause = sum(pauses) / len(pauses) if pauses else 0

        # Calculate average word duration
        word_durations = []
        for segment in segments:
            for word in segment.words:
                word_durations.append(word.end - word.start)

        avg_word_duration = sum(word_durations) / len(word_durations) if word_durations else 0

        analysis = {
            'total_segments': len(segments),
            'total_words': total_words,
            'total_duration': total_duration,
            'words_per_minute': words_per_minute,
            'average_pause': avg_pause,
            'total_pauses': len(pauses),
            'average_word_duration': avg_word_duration,
            'estimated_reading_level': self._estimate_reading_level(words_per_minute)
        }

        return analysis

    def _estimate_reading_level(self, wpm: float) -> str:
        """Estimate speaking speed level"""
        if wpm < 110:
            return "slow"
        elif wpm < 150:
            return "moderate"
        elif wpm < 180:
            return "fast"
        else:
            return "very fast"

    def generate_edit_decision_list(
        self,
        clean_segments: List[CleanSegment],
        output_path: Path
    ) -> Path:
        """
        Generate an Edit Decision List (EDL) file

        Args:
            clean_segments: List of clean segments
            output_path: Path to output EDL file

        Returns:
            Path to EDL file
        """
        try:
            with open(output_path, 'w') as f:
                f.write("TITLE: Video Edit Decision List\n\n")

                for i, segment in enumerate(clean_segments, 1):
                    f.write(f"{i:04d}  001  V  C  ")
                    f.write(f"{self._format_timecode(segment.start)}  ")
                    f.write(f"{self._format_timecode(segment.end)}  ")
                    f.write(f"{self._format_timecode(0)}  ")  # Destination start
                    f.write(f"{self._format_timecode(segment.end - segment.start)}\n")
                    f.write(f"* FROM CLIP: {segment.segment_id}\n")
                    f.write(f"* TEXT: {segment.text}\n\n")

            self.logger.info(f"Generated EDL: {output_path}")
            return output_path

        except Exception as e:
            raise ValidationError(f"Failed to generate EDL: {e}")

    def _format_timecode(self, seconds: float) -> str:
        """Format seconds as timecode (HH:MM:SS:FF)"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        frames = int((seconds % 1) * 30)  # Assuming 30fps

        return f"{hours:02d}:{minutes:02d}:{secs:02d}:{frames:02d}"

    def export_cleaned_transcript(
        self,
        clean_segments: List[CleanSegment],
        output_path: Path
    ) -> Path:
        """
        Export cleaned transcript to text file

        Args:
            clean_segments: List of clean segments
            output_path: Path to output file

        Returns:
            Path to output file
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for segment in clean_segments:
                    f.write(f"[{segment.start:.2f}s - {segment.end:.2f}s] {segment.text}\n")

            self.logger.info(f"Exported cleaned transcript: {output_path}")
            return output_path

        except Exception as e:
            raise ValidationError(f"Failed to export transcript: {e}")
