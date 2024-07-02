#!/bin/sh
uv pip compile --no-cache --python-version=3.11 requirements.in ../pyproject.toml > requirements.txt
