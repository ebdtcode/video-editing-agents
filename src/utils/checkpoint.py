"""
Checkpoint management for resumable processing
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from src.exceptions import CheckpointError
from src.utils.logger import get_logger


logger = get_logger(__name__)


class CheckpointManager:
    """Manage processing checkpoints for resume capability"""

    def __init__(self, checkpoint_file: Path):
        """
        Initialize checkpoint manager

        Args:
            checkpoint_file: Path to checkpoint JSON file
        """
        self.checkpoint_file = Path(checkpoint_file)
        self.state: Dict[str, Any] = self._load()

    def _load(self) -> Dict[str, Any]:
        """Load checkpoint state from file"""
        if not self.checkpoint_file.exists():
            return {
                'version': '1.0',
                'created_at': datetime.now().isoformat(),
                'segments': {},
                'metadata': {}
            }

        try:
            with open(self.checkpoint_file, 'r') as f:
                state = json.load(f)
                logger.info(f"Loaded checkpoint from {self.checkpoint_file}")
                return state
        except Exception as e:
            raise CheckpointError(f"Failed to load checkpoint: {e}")

    def _save(self):
        """Save checkpoint state to file"""
        try:
            self.checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            self.state['updated_at'] = datetime.now().isoformat()

            with open(self.checkpoint_file, 'w') as f:
                json.dump(self.state, f, indent=2)

            logger.debug(f"Saved checkpoint to {self.checkpoint_file}")
        except Exception as e:
            raise CheckpointError(f"Failed to save checkpoint: {e}")

    def mark_segment_started(self, segment_id: str, metadata: Optional[Dict] = None):
        """
        Mark a segment as started

        Args:
            segment_id: Unique segment identifier
            metadata: Optional metadata to store
        """
        self.state['segments'][segment_id] = {
            'status': 'started',
            'started_at': datetime.now().isoformat(),
            'metadata': metadata or {}
        }
        self._save()

    def mark_segment_completed(self, segment_id: str, output_files: Optional[Dict] = None):
        """
        Mark a segment as completed

        Args:
            segment_id: Unique segment identifier
            output_files: Optional dict of output file paths
        """
        if segment_id not in self.state['segments']:
            self.state['segments'][segment_id] = {}

        self.state['segments'][segment_id].update({
            'status': 'completed',
            'completed_at': datetime.now().isoformat(),
            'output_files': output_files or {}
        })
        self._save()

    def mark_segment_failed(self, segment_id: str, error: str):
        """
        Mark a segment as failed

        Args:
            segment_id: Unique segment identifier
            error: Error message
        """
        if segment_id not in self.state['segments']:
            self.state['segments'][segment_id] = {}

        self.state['segments'][segment_id].update({
            'status': 'failed',
            'failed_at': datetime.now().isoformat(),
            'error': error
        })
        self._save()

    def is_segment_completed(self, segment_id: str) -> bool:
        """
        Check if a segment is already completed

        Args:
            segment_id: Unique segment identifier

        Returns:
            True if segment is completed
        """
        return (segment_id in self.state['segments'] and
                self.state['segments'][segment_id].get('status') == 'completed')

    def is_segment_failed(self, segment_id: str) -> bool:
        """
        Check if a segment has previously failed

        Args:
            segment_id: Unique segment identifier

        Returns:
            True if segment is marked as failed
        """
        return (segment_id in self.state['segments'] and
                self.state['segments'][segment_id].get('status') == 'failed')

    def get_segment_status(self, segment_id: str) -> Optional[str]:
        """
        Get segment status

        Args:
            segment_id: Unique segment identifier

        Returns:
            Status string or None if not found
        """
        return self.state['segments'].get(segment_id, {}).get('status')

    def get_completed_segments(self) -> list:
        """Get list of completed segment IDs"""
        return [
            seg_id for seg_id, data in self.state['segments'].items()
            if data.get('status') == 'completed'
        ]

    def get_failed_segments(self) -> list:
        """Get list of failed segment IDs"""
        return [
            seg_id for seg_id, data in self.state['segments'].items()
            if data.get('status') == 'failed'
        ]

    def set_metadata(self, key: str, value: Any):
        """Set checkpoint metadata"""
        self.state['metadata'][key] = value
        self._save()

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get checkpoint metadata"""
        return self.state['metadata'].get(key, default)

    def clear(self):
        """Clear all checkpoint data"""
        self.state = {
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'segments': {},
            'metadata': {}
        }
        self._save()
        logger.info("Checkpoint cleared")

    def get_progress(self) -> Dict[str, int]:
        """
        Get processing progress summary

        Returns:
            Dict with counts of completed, failed, and total segments
        """
        total = len(self.state['segments'])
        completed = len(self.get_completed_segments())
        failed = len(self.get_failed_segments())

        return {
            'total': total,
            'completed': completed,
            'failed': failed,
            'pending': total - completed - failed
        }
