#!/bin/bash
cd $HOME/python/tide

# Check if extract_parallel.py is still running
if pgrep -f "extract_parallel.py" >/dev/null; then
    echo "extract_parallel.py is still running. Scheduling a run in 30 minutes."
    sleep 1800  # Sleep for 30 minutes (30 minutes * 60 seconds)
    /home/bioer/.pyenv/versions/py311/bin/python extract_parallel.py
else
    echo "extract_parallel.py is not running. Starting a new run."
    /home/bioer/.pyenv/versions/py311/bin/python extract_parallel.py
fi
