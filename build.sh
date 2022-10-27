#!/bin/sh

set -eu

CWD=$(basename "$PWD")

build() {
    docker build . --tag "$CWD"
}

clean() {
    docker system prune -f
}

dev() {
    mkdir -p output
    docker run --rm --gpus=all -p 8080:8080 --entrypoint=bash \
        --name "$CWD" \
        -v huggingface:/home/huggingface/.cache/huggingface \
        -v "$PWD"/output:/home/huggingface/output \
        -it "$CWD"
}

run() {
    mkdir -p output
    docker run -d --rm --gpus=all -p 8080:8080 \
        --name "$CWD" \
        -v huggingface:/home/huggingface/.cache/huggingface \
        -v "$PWD"/output:/home/huggingface/output \
        "$CWD"
}

case ${1:-build} in
    build) build ;;
    clean) clean ;;
    dev) dev "$@" ;;
    run) run ;;
    *) echo "$0: No command named '$1'" ;;
esac
