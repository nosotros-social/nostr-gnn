#!/usr/bin/env bash
set -euo pipefail

BLOSSOM_SERVER="${BLOSSOM_SERVER:-https://blossom.nosotros.app}"
MODEL_DIR="${MODEL_DIR:-outputs/modal}"
PROCESSED_DIR="${PROCESSED_DIR:-outputs/modal}"
PRIVATE_KEY_FILE="${PRIVATE_KEY_FILE:-}"

MODEL_PT_PATH="${MODEL_PT_PATH:-$MODEL_DIR/model.pt}"
MODEL_BIN_PATH="${MODEL_BIN_PATH:-$MODEL_DIR/model.bin}"
EMBEDDINGS_PATH="${EMBEDDINGS_PATH:-$MODEL_DIR/embeddings.pt}"
INDEX_NODE_ID_PATH="${INDEX_NODE_ID_PATH:-$PROCESSED_DIR/index_node_id.npy}"
NODE_ID_PUBKEY_PATH="${NODE_ID_PUBKEY_PATH:-$PROCESSED_DIR/node_id_pubkey.parquet}"

if [[ -z "$PRIVATE_KEY_FILE" ]]; then
  echo "PRIVATE_KEY_FILE is required" >&2
  exit 1
fi

PRIVATE_KEY="$(tr -d '\n\r' < "$PRIVATE_KEY_FILE")"

echo "Uploading model.pt: $MODEL_PT_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$MODEL_PT_PATH"
echo "Uploading model.bin: $MODEL_BIN_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$MODEL_BIN_PATH"
echo "Uploading embeddings.pt: $EMBEDDINGS_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$EMBEDDINGS_PATH"
echo "Uploading index_node_id.npy: $INDEX_NODE_ID_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$INDEX_NODE_ID_PATH"
echo "Uploading node_id_pubkey.parquet: $NODE_ID_PUBKEY_PATH"
nak blossom --server "$BLOSSOM_SERVER" --sec "$PRIVATE_KEY" upload "$NODE_ID_PUBKEY_PATH"
