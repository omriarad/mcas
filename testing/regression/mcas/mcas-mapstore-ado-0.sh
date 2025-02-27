#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

TESTID=$(basename --suffix .sh -- $0)

# parameters for MCAS server and client
NODE_IP="$(node_ip)"
DEBUG=${DEBUG:-0}

# launch MCAS server
CONFIG_STR="$("./dist/testing/cfg_mapstore_ado.py" "$NODE_IP")"
[ 0 -lt $DEBUG ] && echo ./dist/bin/mcas --config \'"$CONFIG_STR"\' --forced-exit --debug $DEBUG
./dist/bin/mcas --config "$CONFIG_STR" --forced-exit --debug $DEBUG &> test$TESTID-server.log &
SERVER_PID=$!

# give time to start server
sleep 3

CLIENT_LOG="test$TESTID-client.log"
# launch client
[ 0 -lt $DEBUG ] && echo ./dist/bin/ado-test --src_addr "$NODE_IP" --server $NODE_IP
./dist/bin/ado-test --src_addr "$NODE_IP" --server $NODE_IP &> $CLIENT_LOG &
CLIENT_PID=$!

# arm cleanup
trap "kill -9 $SERVER_PID $CLIENT_PID &> /dev/null" EXIT

# wait for client to complete
wait $CLIENT_PID; CLIENT_RC=$?
wait $SERVER_PID; SERVER_RC=$?

pass_fail_by_code client $CLIENT_RC && pass_fail $CLIENT_LOG $TESTID
