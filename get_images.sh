#!/bin/bash

# Secure the json file from kaggle
chmod 0600 .kaggle/kaggle.json]

# Download the dataset
kaggle datasets download djilax/pkmn-image-dataset

# Check if a zip file was provided
if [ -z "$1" ]
then
    echo "No zip file detected, at least one zip file is needed for unzipping."
    exit 1
fi

# Define the zip file and the temporary directory as arguments
zipfile="$1"
outputdir="${2:-.}"

# Unzip the file to the temporary directory
unzip -q "$zipfile" -d "$outputdir"

# Remove the zip file
rm "$zipfile"