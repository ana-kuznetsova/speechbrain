#!/bin/bash

# Source and destination directories
SOURCE_DIR="/data/anakuzne/LibriSpeech"
DEST_DIR="/data/anakuzne/datasets/LS250BPS"

# Find all .trans.txt files in the source directory
find "$SOURCE_DIR" -type f -name "*.trans.txt" | while read -r file; do
    # Get the relative path of the file
    RELATIVE_PATH="${file#$SOURCE_DIR/}"
    
    # Construct the destination path
    DEST_PATH="$DEST_DIR/$RELATIVE_PATH"
    
    # Ensure the destination directory exists
    mkdir -p "$(dirname "$DEST_PATH")"
    
    # Copy the file to the destination
    cp "$file" "$DEST_PATH"
    
    echo "Copied: $file -> $DEST_PATH"
done
