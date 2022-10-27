#!/usr/bin/python3
# -*- coding: utf-8 -*-
import argparse
import requests
from urllib.parse import quote

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--skip', action='store_true')
parser.add_argument('--steps', type=int, default=20)
parser.add_argument('--timeout', type=int, default=30)
parser.add_argument('--url', type=str, default=None)
parser.add_argument('args', nargs='+', default='')
args = parser.parse_args()
if args.skip:
    param = '?nsfw'
    if args.url:
        param += '&url={}'.format(args.url)
else:
    param = ''
    if args.url:
        param += '?url={}'.format(args.url)

prompt = quote(' '.join(args.args))
if args.seed:
    requests.get('http://localhost:8080/{}/{}/{}{}'.format(prompt, args.steps, args.seed, param), timeout=args.timeout)
else:
    requests.get('http://localhost:8080/{}/{}{}'.format(prompt, args.steps, param), timeout=args.timeout)
