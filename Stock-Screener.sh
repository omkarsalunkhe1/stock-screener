#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
export PYTHONPATH="$SCRIPT_DIR:$PYTHONPATH"

echo "Starting options-gpt server..."
echo "Directory: $SCRIPT_DIR"
echo "Access at: http://localhost:8000"
echo ""

# Try to open Chrome on Windows
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
    CHROME_PATHS=(
        "/c/Program Files/Google/Chrome/Application/chrome.exe"
        "/c/Program Files (x86)/Google/Chrome/Application/chrome.exe"
        "$LOCALAPPDATA/Google/Chrome/Application/chrome.exe"
    )
    
    for chrome in "${CHROME_PATHS[@]}"; do
        if [ -f "$chrome" ]; then
            echo "Opening Chrome..."
            "$chrome" "http://localhost:8000" &
            break
        fi
    done
fi

# Start the server
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload