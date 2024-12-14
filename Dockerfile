# Base has Python with pip
FROM debian:12-slim AS dev-base

WORKDIR /app

RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-pip gcc libpython3-dev


# Compile is meant to generate requirements for the last installable step
FROM dev-base AS compile

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=secret,id=pip_index,env=PIP_INDEX_URL \
    --mount=type=secret,id=pip_index,env=PIP_EXTRA_INDEX_URL \
    pip install uv==0.5.9 \
    --break-system-packages \
    --disable-pip-version-check

COPY pyproject.toml uv.lock /app/

# We want the same version of packages, but we might need more
# because the lock file might be meant for another architecture or version of Python
RUN uv export --format requirements-txt --output-file /constraints.txt \
    --no-editable --no-dev --no-emit-workspace --frozen \
    --no-index --no-hashes

# We then compile using the constraints and our python we expect to run in production
RUN uv pip compile --constraints /constraints.txt --output-file /requirements.txt pyproject.toml

FROM dev-base AS install

COPY --from=compile /requirements.txt /requirements.txt

# Install everything we need into /app
RUN --mount=type=cache,target=/root/.cache \
    --mount=type=secret,id=pip_index,env=PIP_INDEX_URL \
    --mount=type=secret,id=pip_index,env=PIP_EXTRA_INDEX_URL \
    pip install \
    --no-deps --disable-pip-version-check \
    --target /app \
    --requirement /requirements.txt


ADD dist/*.whl /tmp

RUN --mount=type=cache,target=/root/.cache \
    --mount=type=secret,id=pip_index,env=PIP_INDEX_URL \
    --mount=type=secret,id=pip_index,env=PIP_EXTRA_INDEX_URL \
    pip install \
    --no-deps --disable-pip-version-check \
    --target /app \
    /tmp/*.whl

FROM gcr.io/distroless/python3-debian12:nonroot
COPY --from=install /app /app
WORKDIR /app
ENV PYTHONPATH=/app
ENTRYPOINT [ "/usr/bin/python", "-m", "polario.main" ]
