# Stable Diffusion in Docker with HTTP interface

inspired https://github.com/fboulnois/stable-diffusion-docker

many thx!!

## Before you start

The pipeline uses the full model and weights which requires 8GB+ of GPU RAM.
On smaller GPUs you may need to modify some of the parameters. It should take a
few seconds to create one image.

Since it uses the official model, you will need to create a [user access token](https://huggingface.co/docs/hub/security-tokens)
in your [Huggingface account](https://huggingface.co/settings/tokens). Save the
user access token in a file called `token.txt` and make sure it is available
when building the container.

## Quickstart

The pipeline is managed using a single [`build.sh`](build.sh) script. You must
build the image before it can be run.

## Build

Make sure your [user access token](#before-you-start) is saved in a file called
`token.txt`. The token content should begin with `hf_...`

To build:

```sh
./build.sh build  # or just ./build.sh
```

## Run Docker Container

To run:

```shell
./build.sh run
```

## Outputs

### Model

The model and other files are cached in a volume called `huggingface`.

### Images

The images are saved as PNGs in the `output` folder using the prompt text. The
`build.sh` script creates and mounts this folder as a volume in the container.

## Generate

```shell
./gen.py An impressionist painting of a parakeet eating spaghetti in the desert
```

if your terminal supports sixel, install libsixel-bin (also img2sixel) and run

```shell
./gen.sh An impressionist painting of a parakeet eating spaghetti in the desert
```

### switch model

```shell
./switch-model.sh
```
