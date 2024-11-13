#!/bin/bash
rm -rf dist
uv build
uv pip compile pyproject.toml -o constraints.txt
export PIP_INDEX_URL=https://pypi.org/simple
DOCKER_BUILDKIT=1 docker build \
    --secret id=pip_index_url,env=PIP_INDEX_URL \
    --tag polario \
    .
# After building the container, you could run it against a local parquet file like so:
# docker run -it -v $PWD:$PWD:ro --rm polario show $PWD/sample.parquet
