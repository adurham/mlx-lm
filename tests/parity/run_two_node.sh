#!/bin/sh
# Two-node launcher for the DSv4.1 parity harness (stream P) -- PARENT ONLY.
#
# Runs one harness invocation on both Macs as a JACCL TP=2 pair, exactly the
# way p64_prefill.py was launched (same RDMA matrix / coordinator / reliability
# env). Production must be STOPPED first (the parent's call, not this script's).
#
# Usage:  tests/parity/run_two_node.sh <mode> [harness args...]
#   e.g.  tests/parity/run_two_node.sh record --layers all --max-new 32 \
#             --ref ~/dsv41-ws2/parity/ref_full.json --note "full two-node ref"
#         tests/parity/run_two_node.sh check  --layers all \
#             --ref ~/dsv41-ws2/parity/ref_full.json
#
# Rank 0 = macstudio-m4-2, rank 1 = macstudio-m4-1 (phase-14 layout).
# Logs land in ~/dsv41-ws2/parity/<mode>-r<rank>.log on each node.
# The harness itself sets EXL3_MM_MAX_ROWS / MTL_DISABLE_TIMEOUT; the rest of
# the Metal-timeout env is inherited from the launcher.
set -e
MODE=${1:?usage: run_two_node.sh <mode> [args...]}
shift
NODES_R0="macstudio-m4-2"
NODES_R1="macstudio-m4-1"
PKG="\$HOME/dsv41-ws2/P"
OUTDIR="\$HOME/dsv41-ws2/parity"

launch() {   # launch <rank> <ssh-host> <quoted-args>
    RANK=$1; HOST=$2; ARGS=$3
    if [ "$RANK" = 0 ]; then COORD="0.0.0.0:49231"; else COORD="192.168.201.2:49231"; fi
    ssh "$HOST" "mkdir -p $OUTDIR && cd $PKG && \
        echo '[[null, \"rdma_en3\"], [\"rdma_en4\", null]]' > \$HOME/p47_ibv.json && \
        lockf -k \$HOME/dsv41-gpu.lock env \
            MLX_IBV_DEVICES=\$HOME/p47_ibv.json MLX_RANK=$RANK \
            MLX_JACCL_COORDINATOR=$COORD \
            MLX_JACCL_RELIABLE_DATA=1 MLX_JACCL_RELIABLE_MAX_SZ=2 \
            MLX_JACCL_RELIABLE_INFLIGHT=8 MLX_JACCL_RELIABLE_OPTIMISTIC=1 \
            MLX_JACCL_RECONNECT_FRESH=1 MLX_JACCL_ACK_SYNC_PRE=1 \
            MLX_JACCL_ACK_RETRANSMIT_US=500000 IBV_FORK_SAFE=1 \
            MLX_EVENT_WAIT_TIMEOUT_MS=20000 \
            MTL_COMMAND_BUFFER_TIMEOUT=0 EXO_DISABLE_METAL_TIMEOUT=1 \
            AGX_RELAX_CDM_CTXSTORE_TIMEOUT=1 MLX_MAX_OPS_PER_BUFFER=200 \
            MLX_MAX_MB_PER_BUFFER=200 \
            \$HOME/repos/exo/.venv/bin/python -u tests/parity/dsv41_parity.py \
            $MODE --dist jaccl --pkg-root \$HOME/dsv41-ws2/P $ARGS \
            > $OUTDIR/$MODE-r$RANK.log 2>&1; echo rank$RANK exit=\$?" &
}

# shell-quote the harness args so notes with spaces survive the ssh layer
ARGS=$(printf '%s ' "$@")

launch 0 "$NODES_R0" "$ARGS"
launch 1 "$NODES_R1" "$ARGS"
wait
echo "--- rank 0 tail:"
ssh "$NODES_R0" "tail -3 $OUTDIR/$MODE-r0.log"
echo "--- rank 1 tail:"
ssh "$NODES_R1" "tail -3 $OUTDIR/$MODE-r1.log"
