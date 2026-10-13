#!/bin/sh
# Pin all dependencies of the image, including the extras of the package
uv pip compile --no-cache --python-version=3.14 \
    --extra common --extra diffraction --extra gui \
    requirements.in ../pyproject.toml > requirements.txt
