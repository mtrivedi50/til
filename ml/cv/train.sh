#!/bin/bash

# Usage:
#     /bin/sh train.sh -u ubuntu -h 192.222.58.119 -i ~/.ssh/lambda_test.pem -c
#
# Description:
#     Copies data and trains a CNN model on a remote GPU instance.
#
# Options:
#     -u, --username    Remote username
#     -h, --host        Remote host IP
#     -i, --identity    Path to PEM key
#     -c, --copy-data   Copy data before training


while [[ "$#" -gt 0 ]]; do
    case $1 in
        --username|-u)    USERNAME=$2; shift ;;
        --host|-h)        HOST=$2    ; shift ;;
        --identity|-i)    PEM_KEY=$2 ; shift ;;
        --copy-data|-c)   COPY_DATA=true     ;;
        --nnodes)         NNODES=$2  ; shift ;;
        --nproc_per_node) NPROC_PER_NODE=$2  ; shift ;;
    esac
    shift
done

NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
COPY_DATA=${COPY_DATA:-false}

if $COPY_DATA; then
    /bin/sh copy-data.sh -u $USERNAME -h $HOST -i $PEM_KEY
fi

# Copy training script
start=$SECONDS
scp -i ${PEM_KEY} cnn.py ${USERNAME}@${HOST}:~/cnn.py
echo "Copied cnn.py in $(($SECONDS - $start)) seconds"


# Train
/bin/sh setup-gpu.sh -u $USERNAME -h $HOST -i $PEM_KEY
ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "source ~/.venv/bin/activate && torchrun \
    --nnodes=$NNODES \
    --nproc_per_node=$NPROC_PER_NODE \
    cnn.py"
