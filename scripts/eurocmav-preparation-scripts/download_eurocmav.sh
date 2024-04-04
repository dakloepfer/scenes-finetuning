#!/bin/bash

# Function to download a file
download_file() {
  url=$1
  filename=$(basename "$url")
  wget "$url" && echo "Downloaded: $filename" || echo "Failed to download: $filename"
}

mkdir euroc-mav
cd euroc-mav

# Array of download links
download_links=(
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_01_easy/MH_01_easy.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_03_medium/MH_03_medium.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_04_difficult/MH_04_difficult.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_05_difficult/MH_05_difficult.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_01_easy/V1_01_easy.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_02_medium/V1_02_medium.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room1/V1_03_difficult/V1_03_difficult.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_01_easy/V2_01_easy.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_02_medium/V2_02_medium.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/vicon_room2/V2_03_difficult/V2_03_difficult.zip"
  "http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/calibration_datasets/"
)

# Maximum parallel downloads
max_parallel=7

# Download files in parallel with a maximum limit
index=0
num_links=${#download_links[@]}
while ((index < num_links)); do
  for ((count = 0; count < max_parallel && index < num_links; count++, index++)); do
    download_file "${download_links[index]}" &
  done
  wait
done

# Iterate through all files in the current directory
for file in *; do
  if [[ -f "$file" && "$file" == *.zip ]]; then
    # Create a new subfolder with the same name as the .zip file
    folder="${file%.zip}"
    mkdir "$folder"

    # Unzip the .zip file
    unzip -d "$folder" "$file"

    # Delete the .zip file
    rm "$file"
  fi
done