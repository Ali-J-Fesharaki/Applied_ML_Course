#!/bin/bash

# Remove .venv directories in subdirectories
find . -type d -name .venv -exec rm -rf {} +

# Add .venv/ to .gitignore if it's not already there
if ! grep -q "^\.venv/$" .gitignore; then
    echo ".venv/" >> .gitignore
    echo "Added .venv/ to .gitignore"
else
    echo ".venv/ is already in .gitignore"
fi