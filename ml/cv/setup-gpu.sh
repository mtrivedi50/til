#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --username|-u)  USERNAME=$2; shift ;;
        --host|-h)      HOST=$2    ; shift ;;
        --identity|-i)  PEM_KEY=$2 ; shift ;;
    esac
    shift
done

VENV_EXISTS=$(ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "test -f /home/${USERNAME}/pyproject.toml && echo yes || echo no")
if [[ $VENV_EXISTS == "no" ]]; then
    ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "curl -LsSf https://astral.sh/uv/install.sh | sh"
    ssh -i ${PEM_KEY} ${USERNAME}@${HOST} "
        alias uv="~/.local/bin/uv"
        uv init . &&
        uv python install 3.13 &&
        uv python pin 3.13 &&
        uv add pydantic torchvision torch tensorboard &&
        uv add torchcodec --index=https://download.pytorch.org/whl/cu130"
fi
