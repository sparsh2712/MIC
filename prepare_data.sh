#!/bin/bash

# Define paths
DATA_DIR="data"
CLEAN_DIR="${DATA_DIR}/clean_imgs"
NOISY_DIR="${DATA_DIR}/noisy_imgs"
TEST_CLEAN_DIR="${DATA_DIR}/dl_train_imgs/clean_imgs"
TEST_NOISY_DIR="${DATA_DIR}/dl_train_imgs/noisy_imgs"

# Create test directories if they don't exist
mkdir -p ${TEST_CLEAN_DIR}
mkdir -p ${TEST_NOISY_DIR}

# Get all noise types
NOISE_TYPES=$(ls ${NOISY_DIR} | sed 's/.*_\(.*\)\.png/\1/' | sort | uniq)

# Process each noise type
for noise_type in ${NOISE_TYPES}; do
    # Find files with current noise type
    files=($(find ${NOISY_DIR} -name "*_${noise_type}.png"))
    
    # Select 100 files or all if less than 100
    count=0
    max_files=1000
    if [ ${#files[@]} -lt ${max_files} ]; then
        max_files=${#files[@]}
    fi
    
    # Move files
    for file in ${files[@]}; do
        if [ ${count} -ge ${max_files} ]; then
            break
        fi
        
        # Extract base name (number)
        base_name=$(basename ${file} | cut -d'_' -f1)
        clean_file="${CLEAN_DIR}/${base_name}.png"
        
        # Move files only if clean file exists
        if [ -f ${clean_file} ]; then
            # Move noisy image
            mv ${file} ${TEST_NOISY_DIR}/
            
            # Move corresponding clean image
            mv ${clean_file} ${TEST_CLEAN_DIR}/
            
            count=$((count + 1))
        fi
    done
    
    echo "Moved ${count} files with noise type: ${noise_type}"
done