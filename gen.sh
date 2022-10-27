#!/bin/bash
# -*- coding: utf-8 -*-

prompt=$*
s512=$(echo -n "$*" | sha512sum | cut -c-32)
./gen.py --skip $prompt
f=$(ls -1t output/${s512}__* 2> /dev/null | head -1)
test -f $f && img2sixel $f && echo $f
