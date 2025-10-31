#!/usr/bin/env python3
"""
Auto-correct transcript using LLM (OpenAI, Anthropic, or Ollama).

Takes the output from export_transcript_for_editing.py and automatically
corrects grammar, fixes transcription errors, and improves clarity using AI.

Usage:
    # Using OpenAI (default)
    python auto_correct_transcript.py --input editable_transcript.json
    
    # Using Anthropic Claude
    python auto_correct_transcript.py --input editable_transcript.json --provider anthropic
    
    # Using local Ollama
    python auto_correct_transcript.py --input editable_transcript.json --provider ollama --model mistral
    
    # Custom output filename
    python auto_correct_transcript.py --input editable_transcript.json --output corrected.json
"""

import json
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


def get_correction_prompt(original_text: str) -> str:
    """Generate the prompt for LLM correction."""
    return f"""Fix transcription errors, grammar mistakes, and improve clarity in this spoken text.

Original transcription:
"{original_text}"

Requirements:
1. Fix obvious transcription errors (e.g., "will review" → "we review")
2. Correct grammar and punctuation
3. Remove filler words (um, uh, etc.) if present
4. Improve clarity while maintaining the original meaning
5. Keep the conversational tone
6. Return ONLY the corrected text, no explanations

Corrected text:"""


def correct_with_openai(text: str, model: str = "gpt-4o-mini", api_key: Optional[str] = None) -> str:
    """Correct text using OpenAI API."""
    try:
        import openai
    except ImportError:
        raise ImportError("OpenAI package not installed. Install with: pip install openai")
    
    if api_key:
        client = openai.OpenAI(api_key=api_key)
    else:
        client = openai.OpenAI()  # Uses OPENAI_API_KEY env var
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a professional transcript editor. Fix transcription errors and improve text clarity while maintaining the original meaning and conversational tone."},
            {"role": "user", "content": get_correction_prompt(text)}
        ],
        temperature=0.3,
        max_tokens=500
    )
    
    return response.choices[0].message.content.strip()


def correct_with_anthropic(text: str, model: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None) -> str:
    """Correct text using Anthropic Claude API."""
    try:
        import anthropic
    except ImportError:
        raise ImportError("Anthropic package not installed. Install with: pip install anthropic")
    
    if api_key:
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
    
    message = client.messages.create(
        model=model,
        max_tokens=500,
        temperature=0.3,
        system="You are a professional transcript editor. Fix transcription errors and improve text clarity while maintaining the original meaning and conversational tone.",
        messages=[
            {"role": "user", "content": get_correction_prompt(text)}
        ]
    )
    
    return message.content[0].text.strip()


def correct_with_ollama(text: str, model: str = "mistral", host: str = "http://localhost:11434") -> str:
    """Correct text using local Ollama."""
    try:
        import requests
    except ImportError:
        raise ImportError("Requests package not installed. Install with: pip install requests")
    
    url = f"{host}/api/generate"
    
    payload = {
        "model": model,
        "prompt": get_correction_prompt(text),
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 500
        }
    }
    
    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result.get("response", "").strip()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(f"Could not connect to Ollama at {host}. Is Ollama running?")
    except Exception as e:
        raise Exception(f"Ollama request failed: {e}")


def auto_correct_transcript(
    input_file: Path,
    output_file: Path,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    dry_run: bool = False
):
    """Auto-correct transcript using specified LLM provider."""
    
    # Load input JSON
    if not input_file.exists():
        print(f"❌ Error: Input file not found: {input_file}")
        return False
    
    print(f"Loading transcript from: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    segments = data.get('segments', [])
    if not segments:
        print("❌ Error: No segments found in input file")
        return False
    
    # Determine model
    if model is None:
        model_defaults = {
            'openai': 'gpt-4o-mini',
            'anthropic': 'claude-3-5-sonnet-20241022',
            'ollama': 'mistral'
        }
        model = model_defaults.get(provider, 'gpt-4o-mini')
    
    print(f"\n{'='*80}")
    print(f"Provider: {provider}")
    print(f"Model: {model}")
    print(f"Segments: {len(segments)}")
    print(f"Dry run: {dry_run}")
    print(f"{'='*80}\n")
    
    # Get correction function
    correction_funcs = {
        'openai': correct_with_openai,
        'anthropic': correct_with_anthropic,
        'ollama': correct_with_ollama
    }
    
    if provider not in correction_funcs:
        print(f"❌ Error: Unknown provider '{provider}'. Use: openai, anthropic, or ollama")
        return False
    
    correct_func = correction_funcs[provider]
    
    # Process each segment
    corrected_count = 0
    skipped_count = 0
    error_count = 0
    
    for i, segment in enumerate(segments, 1):
        segment_id = segment.get('segment_id', f'seg_{i}')
        original_text = segment.get('original_text', '')
        
        # Skip if segment is marked to skip
        if segment.get('skip', False):
            print(f"[{i}/{len(segments)}] Skipping {segment_id} (marked as skip)")
            skipped_count += 1
            continue
        
        # Skip empty segments
        if not original_text.strip():
            print(f"[{i}/{len(segments)}] Skipping {segment_id} (empty)")
            skipped_count += 1
            continue
        
        print(f"[{i}/{len(segments)}] Correcting {segment_id}...", end=' ')
        
        try:
            if dry_run:
                corrected_text = f"[DRY RUN] {original_text}"
            else:
                # Call LLM to correct
                if provider == 'openai':
                    corrected_text = correct_func(original_text, model, api_key)
                elif provider == 'anthropic':
                    corrected_text = correct_func(original_text, model, api_key)
                else:  # ollama
                    host = api_key if api_key else "http://localhost:11434"
                    corrected_text = correct_func(original_text, model, host)
            
            # Update segment
            segment['corrected_text'] = corrected_text
            if not segment.get('notes'):
                segment['notes'] = f"Auto-corrected by {provider} ({model})"
            
            corrected_count += 1
            print("✓")
            
            # Show sample
            if i <= 3:
                print(f"  Original:  {original_text[:80]}...")
                print(f"  Corrected: {corrected_text[:80]}...")
                print()
        
        except Exception as e:
            print(f"✗ Error: {e}")
            error_count += 1
            # Keep original text on error
            segment['corrected_text'] = original_text
            segment['notes'] = f"Auto-correction failed: {str(e)[:100]}"
    
    # Update metadata
    data['metadata']['auto_corrected_at'] = datetime.now().isoformat()
    data['metadata']['correction_provider'] = provider
    data['metadata']['correction_model'] = model
    data['metadata']['correction_stats'] = {
        'total_segments': len(segments),
        'corrected': corrected_count,
        'skipped': skipped_count,
        'errors': error_count
    }
    
    # Save output
    if not dry_run:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ Auto-Correction Complete!")
    print(f"{'='*80}")
    print(f"Total segments: {len(segments)}")
    print(f"Corrected: {corrected_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors: {error_count}")
    if not dry_run:
        print(f"\nOutput saved to: {output_file}")
        print(f"\nNext step:")
        print(f"  python regenerate_from_corrections.py --input {output_file}")
    else:
        print(f"\nDry run completed. No file was written.")
    print(f"{'='*80}")
    
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description='Auto-correct transcript using LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Using OpenAI (requires OPENAI_API_KEY env var)
  python auto_correct_transcript.py --input editable_transcript.json
  
  # Using Anthropic Claude (requires ANTHROPIC_API_KEY env var)
  python auto_correct_transcript.py --input editable_transcript.json --provider anthropic
  
  # Using local Ollama
  python auto_correct_transcript.py --input editable_transcript.json --provider ollama
  
  # With custom model
  python auto_correct_transcript.py --input editable_transcript.json --model gpt-4o
  
  # Dry run (test without making changes)
  python auto_correct_transcript.py --input editable_transcript.json --dry-run
"""
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Input JSON file from export_transcript_for_editing.py')
    parser.add_argument('--output', '-o',
                       help='Output JSON file (default: input_corrected.json)')
    parser.add_argument('--provider', '-p', default='openai',
                       choices=['openai', 'anthropic', 'ollama'],
                       help='LLM provider to use (default: openai)')
    parser.add_argument('--model', '-m',
                       help='Model name (default: provider-specific default)')
    parser.add_argument('--api-key', '-k',
                       help='API key for OpenAI/Anthropic, or Ollama host URL')
    parser.add_argument('--dry-run', action='store_true',
                       help='Test run without making actual corrections')
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    
    # Determine output filename
    if args.output:
        output_file = Path(args.output)
    else:
        # Generate output filename: input_corrected.json
        output_file = input_file.parent / f"{input_file.stem}_corrected.json"
    
    try:
        success = auto_correct_transcript(
            input_file=input_file,
            output_file=output_file,
            provider=args.provider,
            model=args.model,
            api_key=args.api_key,
            dry_run=args.dry_run
        )
        return 0 if success else 1
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
