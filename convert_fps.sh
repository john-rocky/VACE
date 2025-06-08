#!/bin/bash
# FPS conversion script for WAN output videos

if [ $# -lt 3 ]; then
    echo "Usage: ./convert_fps.sh <input_video> <target_fps> <output_video>"
    echo "Example: ./convert_fps.sh output.mp4 24 output_24fps.mp4"
    exit 1
fi

INPUT=$1
TARGET_FPS=$2
OUTPUT=$3

echo "Converting $INPUT from 16fps to ${TARGET_FPS}fps..."

if [ $TARGET_FPS -le 30 ]; then
    # Simple frame duplication for lower FPS
    ffmpeg -i "$INPUT" -r $TARGET_FPS "$OUTPUT"
else
    # Motion interpolation for higher FPS
    ffmpeg -i "$INPUT" -filter:v "minterpolate=fps=$TARGET_FPS:mi_mode=mci" "$OUTPUT"
fi

echo "Conversion complete: $OUTPUT"

# Show video info
ffmpeg -i "$OUTPUT" 2>&1 | grep "Video:"