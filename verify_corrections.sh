#!/bin/bash
# Verify LLM Corrections Script
# Quick verification of whether LLM corrected the transcript

set -e

WORK_DIR="${1:-temp_segments}"

echo "========================================="
echo "  LLM Correction Verification"
echo "========================================="
echo
echo "Working directory: $WORK_DIR"
echo

# Check if directory exists
if [ ! -d "$WORK_DIR" ]; then
    echo "✗ Directory not found: $WORK_DIR"
    echo
    echo "This could mean:"
    echo "  - No processing has been run yet"
    echo "  - Intermediate files were cleaned up"
    echo "  - Different temp directory was used"
    echo
    echo "Usage: $0 [work_dir]"
    echo "Example: $0 temp_segments"
    exit 1
fi

echo "✓ Working directory found"
echo

# Check correction report
echo "--- Correction Report ---"
if [ -f "$WORK_DIR/correction_report.txt" ]; then
    echo "✓ Correction report exists"
    echo

    # Extract total corrections
    if grep -q "Total segments corrected:" "$WORK_DIR/correction_report.txt"; then
        TOTAL_LINE=$(grep "Total segments corrected:" "$WORK_DIR/correction_report.txt")
        echo "  $TOTAL_LINE"

        # Extract numbers
        CORRECTED=$(echo "$TOTAL_LINE" | grep -oE '[0-9]+' | head -1)
        if [ "$CORRECTED" -gt 0 ]; then
            echo "  ✓ $CORRECTED segments were modified by LLM"
        else
            echo "  ℹ No segments were modified (transcript was already good)"
        fi
    fi

    echo
    echo "  View full report:"
    echo "    cat $WORK_DIR/correction_report.txt"
    echo "    less $WORK_DIR/correction_report.txt"
else
    echo "ℹ No correction report found"
    echo
    echo "  This means either:"
    echo "  - No corrections were made by the LLM"
    echo "  - Content correction was disabled (--no-correction)"
    echo "  - Processing hasn't reached correction stage yet"
fi

echo

# Check transcript files
echo "--- Transcript Files ---"
CLEANED="$WORK_DIR/cleaned_transcript.txt"
CORRECTED="$WORK_DIR/corrected_transcript.txt"

if [ -f "$CLEANED" ] && [ -f "$CORRECTED" ]; then
    echo "✓ Both transcript files exist"

    # Get file sizes
    SIZE_BEFORE=$(wc -c < "$CLEANED" | tr -d ' ')
    SIZE_AFTER=$(wc -c < "$CORRECTED" | tr -d ' ')

    echo "  Before correction: $SIZE_BEFORE bytes"
    echo "  After correction:  $SIZE_AFTER bytes"

    if [ "$SIZE_BEFORE" != "$SIZE_AFTER" ]; then
        DIFF=$((SIZE_AFTER - SIZE_BEFORE))
        if [ "$DIFF" -gt 0 ]; then
            echo "  ✓ Size increased by $DIFF bytes (text added)"
        else
            echo "  ✓ Size decreased by ${DIFF#-} bytes (text removed)"
        fi
    else
        echo "  ℹ Files are same size (content may differ)"
    fi

    echo

    # Compare content
    if command -v diff &> /dev/null; then
        CHANGED_LINES=$(diff "$CLEANED" "$CORRECTED" 2>/dev/null | grep -c "^<" || echo "0")
        if [ "$CHANGED_LINES" -gt 0 ]; then
            echo "  ✓ $CHANGED_LINES lines were modified"
        else
            echo "  ℹ Files are identical (no changes)"
        fi
    fi

    echo
    echo "  Compare transcripts:"
    echo "    diff -u $CLEANED $CORRECTED"
    echo "    diff -y $CLEANED $CORRECTED | less"

elif [ -f "$CLEANED" ]; then
    echo "ℹ Only cleaned transcript exists (correction may not have run)"
    echo "  File: $CLEANED"
else
    echo "ℹ No transcript files found"
fi

echo

# Check logs
echo "--- Processing Logs ---"
LOG_FILE="video_processing.log"

if [ -f "$LOG_FILE" ]; then
    echo "✓ Log file found: $LOG_FILE"
    echo

    # Check for correction stage
    if grep -q "Content Correction" "$LOG_FILE"; then
        echo "  ✓ Correction stage was run"

        # Extract LLM provider
        if grep -q "Initialized.*client" "$LOG_FILE"; then
            PROVIDER=$(grep "Initialized.*client" "$LOG_FILE" | tail -1)
            echo "  $PROVIDER"
        fi

        # Extract correction count
        if grep -q "Corrected.*segments" "$LOG_FILE"; then
            CORRECTION_LINE=$(grep "Corrected.*segments" "$LOG_FILE" | tail -1)
            echo "  $CORRECTION_LINE"
        fi
    else
        echo "  ℹ No correction stage found in logs"
    fi

    echo
    echo "  View correction logs:"
    echo "    grep -i correction $LOG_FILE"
    echo "    grep -i \"stage 3\" $LOG_FILE"
else
    echo "ℹ No log file found: $LOG_FILE"
fi

echo

# Summary
echo "========================================="
echo "  Summary"
echo "========================================="
echo

if [ -f "$WORK_DIR/correction_report.txt" ]; then
    echo "✓ LLM corrections were made and documented"
    echo
    echo "Next steps:"
    echo "  1. Review: cat $WORK_DIR/correction_report.txt"
    echo "  2. Compare: diff -y $CLEANED $CORRECTED | less"
elif [ -f "$CORRECTED" ]; then
    echo "ℹ Correction ran but no changes were made"
    echo "  (The transcript was already clean)"
else
    echo "⚠ Unable to verify corrections"
    echo
    echo "Possible reasons:"
    echo "  - Processing hasn't completed"
    echo "  - Correction was disabled (--no-correction)"
    echo "  - Intermediate files were cleaned up"
fi

echo
