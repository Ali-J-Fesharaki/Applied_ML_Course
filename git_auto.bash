#!/bin/bash

# Maximum file size in bytes (adjust as needed)
MAX_SIZE=$(10*1048576) # 1 MB

# Pull changes from remote repository
git pull origin main

# Add changes to staging area for files in the current directory only
for file in $(git diff --cached --name-only --diff-filter=ACM); do
    if [ ! -d "$file" ]; then
        git add "$file"
    fi
done

# Check file sizes for files in the current directory only
for file in $(git diff --cached --name-only --diff-filter=ACM); do
    if [ ! -d "$file" ]; then
        SIZE=$(wc -c < "$file")
        if [ "$SIZE" -gt "$MAX_SIZE" ]; then
            echo "Error: File '$file' exceeds the size limit of $MAX_SIZE bytes."
            echo "File has been automatically rejected from staging."
            git reset "$file"  # Unstage the file
            exit 1
        fi
    fi
done

# Get commit message from user input
read -p "Enter commit message: " commit_message

# Append current date and time to the commit message
timestamp=$(date +"%Y-%m-%d %H:%M:%S")
commit_message="$commit_message - $timestamp"

# Commit changes
git commit -m "$commit_message"

# Push changes to remote repository
git push origin main
