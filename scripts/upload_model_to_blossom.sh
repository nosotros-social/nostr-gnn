#!/usr/bin/env bash
set -euo pipefail

BLOSSOM_SERVER="${BLOSSOM_SERVER:-https://blossom.nosotros.app}"
MODEL_DIR="${MODEL_DIR:-outputs/modal}"
PROCESSED_DIR="${PROCESSED_DIR:-data/processed}"
PRIVATE_KEY_FILE="${PRIVATE_KEY_FILE:-}"

MODEL_PATH="${MODEL_PATH:-$MODEL_DIR/model.pt}"
INDEX_NODE_ID_PATH="${INDEX_NODE_ID_PATH:-$PROCESSED_DIR/index_node_id.npy}"
NODE_ID_PUBKEY_PATH="${NODE_ID_PUBKEY_PATH:-$PROCESSED_DIR/node_id_pubkey.parquet}"

if [[ -z "$PRIVATE_KEY_FILE" ]]; then
  echo "PRIVATE_KEY_FILE is required" >&2
  exit 1
fi

PRIVATE_KEY="$(tr -d '\n\r' < "$PRIVATE_KEY_FILE")"

nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$MODEL_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$INDEX_NODE_ID_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$NODE_ID_PUBKEY_PATH"
