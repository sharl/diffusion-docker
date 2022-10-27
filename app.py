#!/usr/bin/env python3
import datetime
import hashlib
import torch
from torch import autocast
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from bottle import route, request, run
from PIL import Image
from io import BytesIO
import requests


model_seed = 0
models = [
    'CompVis/stable-diffusion-v1-4',
    'hakurei/waifu-diffusion'
]
model = models[model_seed]

DEVICE = 'cuda'


with open("token.txt") as f:
    token = f.read().replace("\n", "")


def isodatetime():
    return datetime.datetime.now().isoformat()


def skip_safety_checker(images, *args, **kwargs):
    return images, False


class Diffusions:
    def __init__(self, model):
        if model not in models:
            pass

        # select your VRAM
        dtype = torch.float32
        dtype = torch.float16

        # init text to image
        self.pipe = StableDiffusionPipeline.from_pretrained(
            model, torch_dtype=dtype, use_auth_token=token
        ).to(DEVICE)
        self.pipe.enable_attention_slicing()
        self.safety_checker = self.pipe.safety_checker

        # init image to image
        self.img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model, torch_dtype=dtype, use_auth_token=token
        ).to(DEVICE)
        self.img_pipe.enable_attention_slicing()
        self.img_safety_checker = self.img_pipe.safety_checker

        print("loaded models after:", isodatetime())


    def render(self, prompt, height=512, width=512, steps=50, scale=7.5, seed=None, skip=False):
        print("start rendering    :", isodatetime())

        if seed is None:
            seed = torch.random.seed()
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
        if skip:
            self.pipe.safety_checker = skip_safety_checker
        else:
            self.pipe.safety_checker = self.safety_checker

        with autocast(DEVICE):
            images = self.pipe(
                prompt=prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=scale,
                generator=generator,
            )

        print("ended rendering    :", isodatetime())

        s512 = hashlib.sha512(prompt.encode()).hexdigest()[:32]
        iname = bytes(prompt, 'utf-8')[:100].decode('utf-8', 'ignore').replace(' ', '_')
        for image in images['images']:
            image.save(
                "output/%s__%s__steps_%d__scale_%0.2f__seed_%d.png"
                % (s512, iname, steps, scale, seed)
            )

    def img2img(self, prompt, url=None, steps=50, seed=None, skip=False):
        print("start rendering    :", isodatetime())

        image = None
        if url:
            r = requests.get(url, timeout=30)
            if r and r.status_code == 200:
                image = Image.open(BytesIO(r.content)).convert('RGB')
                image = image.resize((512, 512))

        if seed is None:
            seed = torch.random.seed()
        generator = torch.Generator(device=DEVICE).manual_seed(seed)
        if skip:
            self.img_pipe.safety_checker = skip_safety_checker
        else:
            self.img_pipe.safety_checker = self.img_safety_checker

        with autocast(DEVICE):
            images = self.img_pipe(
                prompt=prompt,
                init_image=image,
                strength=0.75,
                num_inference_steps=steps,
                guidance_scale=7.5,
                generator=generator
            ).images

        print("ended rendering    :", isodatetime())

        s512 = hashlib.sha512(prompt.encode()).hexdigest()[:32]
        iname = bytes(prompt, 'utf-8')[:100].decode('utf-8', 'ignore').replace(' ', '_')
        images[0].save(
            'output/%s__%s__steps_%d__seed_%d.png' % (s512, iname, steps, seed)
        )


diffusion = Diffusions(model)


def change_model():
    global model, model_seed, diffusion

    model_seed = (model_seed + 1) % len(models)
    model = models[model_seed]

    del diffusion
    diffusion = Diffusions(model)
    return model


def check_nsfw():
    return 'nsfw' in dict(request.query)


@route('/')
def hello():
    return change_model()


@route('/<text>')
def main(text):
    print('text', text)
    url = request.query.get('url')
    if url:
        diffusion.img2img(text, url=url, steps=20, skip=check_nsfw())
    else:
        diffusion.render(text, steps=20, skip=check_nsfw())


@route('/<text>/<steps:int>')
def main_steps(text, steps):
    print('text', text)
    url = request.query.get('url')
    if url:
        diffusion.img2img(text, url=url, steps=steps, skip=check_nsfw())
    else:
        diffusion.render(text, steps=steps, skip=check_nsfw())


@route('/<text>/<steps:int>/<seed:int>')
def main_steps_seed(text, steps, seed):
    print('text', text)
    url = request.query.get('url')
    if url:
        diffusion.img2img(text, url=url, steps=steps, seed=seed, skip=check_nsfw())
    else:
        diffusion.render(text, steps=steps, seed=seed, skip=check_nsfw())


run(host='0.0.0.0', port=8080, debug=True)
