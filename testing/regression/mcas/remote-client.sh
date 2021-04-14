#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

NODE_IP="$(node_ip)"
DEBUG=${DEBUG:-0}

# launch client

[ 0 -lt $DEBUG ] && echo $DIR/../bin/kvstore-perf --src_addr $NODE_IP $@
$DIR/../bin/kvstore-perf --src_addr "$NODE_IP" $@
