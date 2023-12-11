# Create input folder if it doesn't exist
New-Item -ItemType Directory -Force -Path "input"

# Function to download and untar data
function Download-And-Untar {
    param(
        [string]$url,
        [string]$folderName
    )

    # Create folder inside input
    New-Item -ItemType Directory -Force -Path "input\$folderName"

    # Download and untar
    Invoke-WebRequest -Uri $url -OutFile "input\$folderName\temp.tgz"
    tar -xzf "input\$folderName\temp.tgz" -C "input\$folderName" --strip-components 1
    Remove-Item -Path "input\$folderName\temp.tgz" -Force
}

# URLs and folder names
$url1 = "https://gebakx.github.io/ihlt/sts/resources/trial.tgz"
$folder1 = "trial"

$url2 = "https://gebakx.github.io/ihlt/sts/resources/train.tgz"
$folder2 = "train"

$url3 = "https://gebakx.github.io/ihlt/sts/resources/test-gold.tgz"
$folder3 = "test"

$url4 = "https://gebakx.github.io/ihlt/sts/resources/task6-submissions.tgz"
$folder4 = "all_submissions"

# Download and untar each dataset
Download-And-Untar $url1 $folder1
Download-And-Untar $url2 $folder2
Download-And-Untar $url3 $folder3
Download-And-Untar $url4 $folder4

Write-Host "Data download and extraction complete."
