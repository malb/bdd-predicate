## -*- docker-image-name: "martinralbrecht/bdd-predicate" -*-

FROM fplll/sagemath-g6k:latest
MAINTAINER Martin Albrecht <martinralbrecht+docker@googlemail.com>

ARG JOBS=2
SHELL ["/bin/bash", "-c"]

COPY requirements.txt .

RUN SAGE_ROOT=`pwd`/sage && \
    export SAGE_ROOT="$SAGE_ROOT" && \
    source "$SAGE_ROOT/local/bin/sage-env" && \
    pip3 install -r requirements.txt
    pip3 install black

RUN rm requirements.txt
