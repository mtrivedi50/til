#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --username|-u) USERNAME=$2; shift ;;
        --host|-h)     HOST=$2    ; shift ;;
        --identity|-i) PEM_KEY=$2 ; shift ;;
    esac
    shift
done

# Compress data
for dir in data/ucf101/*/; do
    name=$(basename "$dir")
    [[ "$name" == "_compressed" ]] && continue
    output="data/_compressed/$name.tar.gz"

    if [[ ! -f $output ]]; then
        echo "Compressing $name..."
        start=$SECONDS
        tar -czf "$output" -C "data/ucf101" "$name" --exclude='._*' --exclude='.DS_Store'
        echo "Finished compressing $name in $(($SECONDS - $start))s"
    else
        echo "Skipping $name (already exists)"
    fi
done

# Copy and uncompress data on GPU
ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "mkdir -p data/_compressed data/ucf101/ data/annotations"
for fpath in data/_compressed/*; do
    fname=$(basename "$fpath")
    exists=$(ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "test -d /home/${USERNAME}/data/ucf101/${fname%.tar.gz}/ && echo yes || echo no")
    if [[ $exists == "no" ]]; then
        start=$SECONDS
        echo "Copying and uncompressing $fname to GPU"
        scp -i ${PEM_KEY} "$fpath" ${USERNAME}@${HOST}:/home/${USERNAME}/data/_compressed/
        ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "tar -xzf /home/${USERNAME}/data/_compressed/${fname} -C /home/${USERNAME}/data/ucf101/ --warning=no-unknown-keyword"


        echo "Finished copying and uncompressing $fname to GPU in $(($SECONDS - $start)) seconds"
    else
        echo "Uncompressed ${fname} already exists!"
    fi
done

# Delete Mac metadata files (prefixed by ._)
ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "find /home/${USERNAME}/data/ucf101 -name '._*' -delete"

# Copy annotations
scp -i ${PEM_KEY} -r data/annotations ${USERNAME}@${HOST}:/home/${USERNAME}/data
