"""
Content Correction Agent - Uses LLM to improve transcript quality
"""

import os
from pathlib import Path
from typing import List, Optional, Dict
from dataclasses import dataclass

from src.config import ContentCorrectionConfig
from src.agents.content_analysis_agent import CleanSegment
from src.exceptions import ValidationError
from src.utils.logger import get_logger


logger = get_logger(__name__)


@dataclass
class CorrectedSegment:
    """Represents a corrected segment"""
    segment_id: str
    original_text: str
    corrected_text: str
    corrections_made: List[str]
    start: float
    end: float


class ContentCorrectionAgent:
    """Uses LLM to correct grammar and improve transcript clarity"""

    def __init__(self, config: ContentCorrectionConfig):
        """
        Initialize content correction agent

        Args:
            config: Content correction configuration
        """
        self.config = config
        self.logger = get_logger(self.__class__.__name__)
        self.client = None

        if self.config.enabled:
            self._initialize_llm_client()

    def _initialize_llm_client(self):
        """Initialize LLM client based on configuration"""
        if self.config.mode == "openai":
            try:
                import openai

                api_key = self.config.api_key or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "No OpenAI API key found. Content correction disabled. "
                        "Set OPENAI_API_KEY environment variable or config.api_key"
                    )
                    self.config.enabled = False
                    return

                self.client = openai.OpenAI(api_key=api_key)
                self.logger.info(f"Initialized OpenAI client with model: {self.config.model}")

            except ImportError:
                self.logger.warning(
                    "OpenAI package not installed. Install with: pip install openai"
                )
                self.config.enabled = False

        elif self.config.mode == "anthropic":
            try:
                import anthropic

                api_key = self.config.api_key or os.getenv("ANTHROPIC_API_KEY")
                if not api_key:
                    self.logger.warning(
                        "No Anthropic API key found. Content correction disabled. "
                        "Set ANTHROPIC_API_KEY environment variable or config.api_key"
                    )
                    self.config.enabled = False
                    return

                self.client = anthropic.Anthropic(api_key=api_key)
                self.logger.info(f"Initialized Anthropic client with model: {self.config.model}")

            except ImportError:
                self.logger.warning(
                    "Anthropic package not installed. Install with: pip install anthropic"
                )
                self.config.enabled = False

        elif self.config.mode == "ollama":
            try:
                # Test Ollama connection
                import requests

                # Default Ollama endpoint
                ollama_host = self.config.api_key or os.getenv("OLLAMA_HOST", "http://localhost:11434")

                # Test connection
                response = requests.get(f"{ollama_host}/api/tags")
                if response.status_code != 200:
                    self.logger.warning(
                        f"Cannot connect to Ollama at {ollama_host}. "
                        "Make sure Ollama is running: ollama serve"
                    )
                    self.config.enabled = False
                    return

                # Store Ollama host
                self.ollama_host = ollama_host
                self.client = "ollama"  # Marker that we're using Ollama

                self.logger.info(
                    f"Initialized Ollama client with model: {self.config.model} "
                    f"at {ollama_host}"
                )

            except ImportError:
                self.logger.warning(
                    "Requests package not installed. Install with: pip install requests"
                )
                self.config.enabled = False
            except Exception as e:
                self.logger.warning(f"Failed to connect to Ollama: {e}")
                self.config.enabled = False

        else:
            self.logger.warning(f"Unsupported LLM mode: {self.config.mode}")
            self.config.enabled = False

    def correct_segments(
        self,
        segments: List[CleanSegment]
    ) -> List[CleanSegment]:
        """
        Correct multiple segments using LLM

        Args:
            segments: List of clean segments

        Returns:
            List of corrected segments (as CleanSegment objects)
        """
        if not self.config.enabled:
            self.logger.info("Content correction disabled, returning original segments")
            return segments

        self.logger.info(f"Correcting {len(segments)} segments with {self.config.mode}")

        corrected_segments = []
        for segment in segments:
            try:
                corrected = self.correct_segment(segment)

                # Create new CleanSegment with corrected text
                corrected_segment = CleanSegment(
                    segment_id=segment.segment_id,
                    text=corrected.corrected_text,
                    start=segment.start,
                    end=segment.end,
                    words=segment.words,
                    original_text=segment.original_text  # Keep the truly original text
                )
                corrected_segments.append(corrected_segment)

                if corrected.corrections_made:
                    self.logger.debug(
                        f"{segment.segment_id}: {len(corrected.corrections_made)} corrections made"
                    )

            except Exception as e:
                self.logger.warning(
                    f"Failed to correct segment {segment.segment_id}: {e}. Using original."
                )
                corrected_segments.append(segment)

        return corrected_segments

    def correct_segment(self, segment: CleanSegment) -> CorrectedSegment:
        """
        Correct a single segment using LLM

        Args:
            segment: Clean segment to correct

        Returns:
            CorrectedSegment with improvements
        """
        if not self.config.enabled:
            return CorrectedSegment(
                segment_id=segment.segment_id,
                original_text=segment.text,
                corrected_text=segment.text,
                corrections_made=[],
                start=segment.start,
                end=segment.end
            )

        # Build correction prompt
        prompt = self._build_correction_prompt(segment.text)

        # Get correction from LLM
        try:
            if self.config.mode == "openai":
                corrected_text = self._correct_with_openai(prompt)
            elif self.config.mode == "anthropic":
                corrected_text = self._correct_with_anthropic(prompt)
            elif self.config.mode == "ollama":
                corrected_text = self._correct_with_ollama(prompt)
            else:
                corrected_text = segment.text

            # Validate the correction response
            if not self._is_valid_correction(segment.text, corrected_text):
                self.logger.warning(
                    f"LLM returned invalid correction for segment {segment.segment_id}, "
                    "using original text"
                )
                corrected_text = segment.text

            # Detect what corrections were made
            corrections_made = self._detect_corrections(segment.text, corrected_text)

            return CorrectedSegment(
                segment_id=segment.segment_id,
                original_text=segment.text,
                corrected_text=corrected_text,
                corrections_made=corrections_made,
                start=segment.start,
                end=segment.end
            )

        except Exception as e:
            self.logger.error(f"LLM correction failed: {e}")
            return CorrectedSegment(
                segment_id=segment.segment_id,
                original_text=segment.text,
                corrected_text=segment.text,
                corrections_made=[],
                start=segment.start,
                end=segment.end
            )

    def _build_correction_prompt(self, text: str) -> str:
        """Build prompt for LLM correction"""
        features = []
        if self.config.fix_grammar:
            features.append("Fix any grammar errors")
        if self.config.rephrase_awkward:
            features.append("Rephrase awkward phrasing")
        if self.config.remove_repetitions:
            features.append("Remove unnecessary repetitions")
        if self.config.improve_clarity:
            features.append("Improve clarity and conciseness")

        features_text = ", ".join(features)

        prompt = f"""You are a transcript editor. Below is a transcript segment that needs correction.

CRITICAL: Output ONLY the corrected text itself. Do NOT include:
- Explanations, notes, or comments
- Phrases like "Here is the corrected text" or "I'm ready to help"
- Questions asking for more information
- Any meta-commentary about the task

Task: {features_text}
- Keep the meaning and intent unchanged
- Make it sound natural when spoken aloud
- Preserve any technical terms or specific names

Transcript to correct:
{text}

Corrected transcript:"""

        return prompt

    def _correct_with_openai(self, prompt: str) -> str:
        """Get correction using OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": "You are a professional transcript editor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            corrected_text = response.choices[0].message.content.strip()
            return corrected_text

        except Exception as e:
            raise ValidationError(f"OpenAI API error: {e}")

    def _correct_with_anthropic(self, prompt: str) -> str:
        """Get correction using Anthropic API"""
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            corrected_text = response.content[0].text.strip()
            return corrected_text

        except Exception as e:
            raise ValidationError(f"Anthropic API error: {e}")

    def _correct_with_ollama(self, prompt: str) -> str:
        """Get correction using Ollama (local LLM)"""
        try:
            import requests

            # Ollama API endpoint
            url = f"{self.ollama_host}/api/generate"

            payload = {
                "model": self.config.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_tokens
                }
            }

            response = requests.post(url, json=payload, timeout=60)

            if response.status_code != 200:
                raise ValidationError(
                    f"Ollama API error: {response.status_code} - {response.text}"
                )

            result = response.json()
            corrected_text = result.get("response", "").strip()

            if not corrected_text:
                raise ValidationError("Ollama returned empty response")

            return corrected_text

        except requests.exceptions.Timeout:
            raise ValidationError("Ollama request timed out (60s)")
        except Exception as e:
            raise ValidationError(f"Ollama API error: {e}")

    def _is_valid_correction(self, original: str, corrected: str) -> bool:
        """
        Validate that LLM response is a valid correction and not an error/help message

        Args:
            original: Original text
            corrected: Corrected text from LLM

        Returns:
            True if valid correction, False otherwise
        """
        # Check if response is empty
        if not corrected or not corrected.strip():
            self.logger.warning("LLM returned empty response")
            return False

        # Check if response is too short (likely not a real correction)
        if len(corrected.strip()) < 10 and len(original) > 20:
            self.logger.warning(f"LLM response too short: {len(corrected)} chars")
            return False

        # Check for common error/help phrases that indicate the LLM misunderstood
        error_phrases = [
            "i'm ready to help",
            "please provide",
            "i'll be happy to assist",
            "i'm sorry",
            "could you please",
            "it seems",
            "it looks like",
            "however, it seems",
            "original transcript segment",
            "i can help you",
            "let me know",
            "feel free to",
            "i'd be happy",
            "here is the corrected",
            "here's the corrected"
        ]

        corrected_lower = corrected.lower()
        for phrase in error_phrases:
            if phrase in corrected_lower:
                self.logger.warning(f"LLM returned error/help message containing: '{phrase}'")
                return False

        # Check if response is drastically longer (probably added explanations)
        word_ratio = len(corrected.split()) / max(len(original.split()), 1)
        if word_ratio > 2.5:
            self.logger.warning(
                f"LLM response too verbose: {len(corrected.split())} words "
                f"vs {len(original.split())} original"
            )
            return False

        # Check if response contains question marks (likely asking for clarification)
        if corrected.count('?') > original.count('?') + 1:
            self.logger.warning("LLM response contains unexpected questions")
            return False

        return True

    def _detect_corrections(self, original: str, corrected: str) -> List[str]:
        """Detect what types of corrections were made"""
        corrections = []

        if original.lower() != corrected.lower():
            corrections.append("text_modified")

        if len(corrected.split()) < len(original.split()):
            corrections.append("condensed")

        if len(corrected.split()) > len(original.split()):
            corrections.append("expanded")

        # Simple grammar detection (basic heuristics)
        if original.count(',') != corrected.count(','):
            corrections.append("punctuation")

        if original[0].islower() and corrected[0].isupper():
            corrections.append("capitalization")

        return corrections if corrections else ["no_changes"]

    def generate_correction_report(
        self,
        original_segments: List[CleanSegment],
        corrected_segments: List[CleanSegment],
        output_path: Path
    ) -> Path:
        """
        Generate a report showing all corrections made

        Args:
            original_segments: Original segments
            corrected_segments: Corrected segments
            output_path: Path to output report

        Returns:
            Path to report file
        """
        try:
            total_corrections = 0
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("Content Correction Report\n")
                f.write("=" * 80 + "\n\n")

                for orig, corr in zip(original_segments, corrected_segments):
                    if orig.text != corr.text:
                        total_corrections += 1
                        f.write(f"Segment: {orig.segment_id}\n")
                        f.write(f"Time: [{orig.start:.2f}s - {orig.end:.2f}s]\n")
                        f.write(f"\nOriginal:\n{orig.text}\n")
                        f.write(f"\nCorrected:\n{corr.text}\n")
                        f.write("-" * 80 + "\n\n")

                f.write(f"\nTotal segments corrected: {total_corrections}/{len(original_segments)}\n")

            self.logger.info(f"Generated correction report: {output_path}")
            return output_path

        except Exception as e:
            raise ValidationError(f"Failed to generate correction report: {e}")
