#!/bin/bash

# Create input folder if it doesn't exist
mkdir -p input

# Function to download and untar data
download_and_untar() {
    local url=$1
    local folder_name=$2

    # Create folder inside input
    mkdir -p "input/$folder_name"

    # Download and untar
    wget -O - "$url" | tar -xz -C "input/$folder_name" --strip-components=1
}

# URLs and folder names
url1="https://gebakx.github.io/ihlt/sts/resources/trial.tgz"
folder1="trial"

url2="https://gebakx.github.io/ihlt/sts/resources/train.tgz"
folder2="train"

url3="https://gebakx.github.io/ihlt/sts/resources/test-gold.tgz"
folder3="test"

url4="https://gebakx.github.io/ihlt/sts/resources/task6-submissions.tgz"
folder4="all_submissions"

# Download and untar each dataset
download_and_untar "$url1" "$folder1"
download_and_untar "$url2" "$folder2"
download_and_untar "$url3" "$folder3"
download_and_untar "$url4" "$folder4"

echo "Data download and extraction complete."
