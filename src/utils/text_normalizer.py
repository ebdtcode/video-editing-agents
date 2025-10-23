"""
Text normalization utilities for natural-sounding transcripts
"""

import re
from typing import List


class TextNormalizer:
    """Normalizes text for natural speech after filler removal"""

    def __init__(self):
        # Common sentence-ending punctuation
        self.sentence_endings = {'.', '!', '?'}

        # Punctuation that should have no space before it
        self.no_space_before = {'.', ',', '!', '?', ';', ':', "'", '"', ')', ']', '}', '%'}

        # Punctuation that should have no space after it
        self.no_space_after = {'(', '[', '{', '"', "'"}

        # Contractions and possessives
        self.contraction_pattern = re.compile(r"\s+'(s|t|re|ve|d|ll|m)\b", re.IGNORECASE)

    def normalize(self, text: str, capitalize_first: bool = True) -> str:
        """
        Normalize text for natural speech

        Args:
            text: Input text
            capitalize_first: Whether to capitalize first letter

        Returns:
            Normalized text
        """
        if not text or not text.strip():
            return ""

        # Step 1: Fix punctuation spacing
        normalized = self._fix_punctuation_spacing(text)

        # Step 2: Fix contractions (e.g., "don 't" -> "don't")
        normalized = self._fix_contractions(normalized)

        # Step 3: Remove extra whitespace
        normalized = self._remove_extra_whitespace(normalized)

        # Step 4: Capitalize first letter if requested
        if capitalize_first:
            normalized = self._capitalize_first_letter(normalized)

        # Step 5: Ensure proper sentence ending
        normalized = self._ensure_sentence_ending(normalized)

        return normalized

    def _fix_punctuation_spacing(self, text: str) -> str:
        """Fix spacing around punctuation"""
        result = text

        # Remove space before punctuation
        for punct in self.no_space_before:
            # Handle cases like "hello ." -> "hello."
            result = result.replace(f' {punct}', punct)
            # Handle multiple spaces
            result = result.replace(f'  {punct}', punct)

        # Ensure space after punctuation (except for special cases)
        for punct in {'.', ',', '!', '?', ';', ':'}:
            # Add space after punctuation if not already there
            # But don't add at end of string
            result = re.sub(f'\\{punct}([A-Za-z])', f'{punct} \\1', result)

        return result

    def _fix_contractions(self, text: str) -> str:
        """Fix contractions like "don 't" -> "don't" """
        # Fix common contractions
        result = self.contraction_pattern.sub(r"'\1", text)

        # Fix possessives like "John 's" -> "John's"
        result = re.sub(r"\b(\w+)\s+'s\b", r"\1's", result)

        return result

    def _remove_extra_whitespace(self, text: str) -> str:
        """Remove extra whitespace"""
        # Replace multiple spaces with single space
        result = re.sub(r'\s+', ' ', text)

        # Strip leading/trailing whitespace
        result = result.strip()

        return result

    def _capitalize_first_letter(self, text: str) -> str:
        """Capitalize the first letter of text"""
        if not text:
            return text

        # Find first alphabetic character and capitalize it
        for i, char in enumerate(text):
            if char.isalpha():
                return text[:i] + char.upper() + text[i+1:]

        return text

    def _ensure_sentence_ending(self, text: str) -> str:
        """Ensure text ends with proper punctuation"""
        if not text:
            return text

        # Check if already ends with sentence-ending punctuation
        if text[-1] in self.sentence_endings:
            return text

        # Check if ends with other punctuation (like comma)
        if text[-1] in {',', ';', ':'}:
            # Replace with period
            return text[:-1] + '.'

        # No punctuation at end - add period
        return text + '.'

    def normalize_segment_text(self, text: str, is_first_segment: bool = False) -> str:
        """
        Normalize text for a segment

        Args:
            text: Input segment text
            is_first_segment: Whether this is the first segment

        Returns:
            Normalized segment text
        """
        # Always capitalize first letter of each segment
        normalized = self.normalize(text, capitalize_first=True)

        return normalized

    def join_words_naturally(self, words: List[str]) -> str:
        """
        Join words with proper spacing and punctuation

        Args:
            words: List of words (may include punctuation attached)

        Returns:
            Naturally joined text
        """
        if not words:
            return ""

        # Join with spaces
        text = " ".join(words)

        # Normalize
        return self.normalize(text)

    def clean_transcript_line(self, text: str) -> str:
        """
        Clean a line from transcript for natural speech

        This is useful for cleaning individual segment text
        before TTS generation.

        Args:
            text: Raw transcript text

        Returns:
            Cleaned text ready for TTS
        """
        if not text:
            return text

        # Remove multiple punctuation (e.g., "..." -> ".", "!!!" -> "!")
        text = re.sub(r'\.{2,}', '.', text)
        text = re.sub(r'\?{2,}', '?', text)
        text = re.sub(r'!{2,}', '!', text)

        # Remove special characters that TTS might struggle with
        # Keep: letters, numbers, spaces, basic punctuation
        text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)

        # Normalize spacing and capitalization
        # Note: normalize() adds ending punctuation if missing
        text = self.normalize(text, capitalize_first=True)

        return text


def normalize_text(text: str, capitalize: bool = True) -> str:
    """
    Convenience function for text normalization

    Args:
        text: Input text
        capitalize: Whether to capitalize first letter

    Returns:
        Normalized text
    """
    normalizer = TextNormalizer()
    return normalizer.normalize(text, capitalize_first=capitalize)


def normalize_segment_list(segments: List[str]) -> List[str]:
    """
    Normalize a list of segment texts

    Args:
        segments: List of segment texts

    Returns:
        List of normalized texts
    """
    normalizer = TextNormalizer()
    return [
        normalizer.normalize_segment_text(seg, is_first=(i == 0))
        for i, seg in enumerate(segments)
    ]
