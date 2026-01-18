#!/usr/bin/env bash
EDGE="$1"
CLUST="$2"
OUT="$3"
NUM="${4:-1}"

docker compose --profile ecsbm run --rm ecsbm bash docker/ecsbm/run.sh "$EDGE" "$CLUST" "$OUT" "$NUM"
