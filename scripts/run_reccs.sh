#!/usr/bin/env bash
EDGE="$1"
CLUST="$2"
OUT="$3"
NUM="${4:-1}"

docker compose --profile reccs run --rm reccs bash docker/reccs/run.sh "$EDGE" "$CLUST" "$OUT" "$NUM"
