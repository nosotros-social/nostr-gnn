#!/usr/bin/env bash
set -euo pipefail

OUTPUT_VOLUME_NAME="${OUTPUT_VOLUME_NAME:-nostr-gnn-outputs}"
DATA_VOLUME_NAME="${DATA_VOLUME_NAME:-nostr-gnn-data}"
REMOTE_OUTPUT_PATH="${REMOTE_OUTPUT_PATH:-/modal}"
LOCAL_OUTPUT_PATH="${LOCAL_OUTPUT_PATH:-outputs}"
REMOTE_PROCESSED_PATH="${REMOTE_PROCESSED_PATH:-/processed}"
LOCAL_PROCESSED_PATH="${LOCAL_PROCESSED_PATH:-outputs/modal}"

modal volume get "$OUTPUT_VOLUME_NAME" "$REMOTE_OUTPUT_PATH" "$LOCAL_OUTPUT_PATH" --force
modal volume get "$DATA_VOLUME_NAME" "$REMOTE_PROCESSED_PATH/index_node_id.npy" "$LOCAL_PROCESSED_PATH/index_node_id.npy" --force
modal volume get "$DATA_VOLUME_NAME" "$REMOTE_PROCESSED_PATH/node_id_pubkey.parquet" "$LOCAL_PROCESSED_PATH/node_id_pubkey.parquet" --force
