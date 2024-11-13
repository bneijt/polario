FROM debian:12-slim AS build
RUN apt-get update && \
    apt-get install --no-install-suggests --no-install-recommends --yes python3-pip gcc libpython3-dev

ADD dist/*.whl /tmp

RUN --mount=type=bind,source=constraints.txt,target=/tmp/constraints.txt \
    --mount=type=secret,id=pip_index,env=PIP_INDEX_URL \
    --mount=type=secret,id=pip_index,env=PIP_EXTRA_INDEX_URL \
    pip install \
        --target /app \
        --disable-pip-version-check \
        --constraint /tmp/constraints.txt \
        /tmp/*.whl 

FROM gcr.io/distroless/python3
COPY --from=build /app /app
WORKDIR /app
ENV PYTHONPATH=/app
ENTRYPOINT [ "/usr/bin/python", "-m", "polario.main" ]
