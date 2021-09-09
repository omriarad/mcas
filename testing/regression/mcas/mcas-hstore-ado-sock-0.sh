#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

DAX_PREFIX="${DAX_PREFIX:-$(choose_dax)}"
STORETYPE=hstore
TESTID="$(basename --suffix .sh -- $0)-$(dax_type $DAX_PREFIX)"

# parameters for MCAS server and client
NODE_IP="$(node_ip)"
DEBUG=${DEBUG:-0}

$CONFIG_STR="$("./dist/testing/hstore-ado0-sock.py" "$STORETYPE" "$DAX_PREFIX" "$NODE_IP" 11911 )"
NUMA_CMD=$(numa_cmd $DAX_PREFIX)
# launch MCAS server
[ 0 -lt $DEBUG ] && echo DAX_RESET=1 ${NUMA_CMD} ./dist/bin/mcas --config \'"$CONFIG_STR"\' --forced-exit --debug $DEBUG
DAX_RESET=1 ${NUMA_CMD} ./dist/bin/mcas --config "$CONFIG_STR" --forced-exit --debug $DEBUG &> test$TESTID-server.log &
SERVER_PID=$!

# give time to start server
sleep 3

CLIENT_LOG="test$TESTID-client.log"

# launch client
[ 0 -lt $DEBUG ] && echo ./dist/bin/ado-test --provider sockets --src_addr "$NODE_IP" --server $NODE_IP --port 11911
./dist/bin/ado-test --provider sockets --src_addr "$NODE_IP" --server $NODE_IP --port 11911 &> $CLIENT_LOG &
CLIENT_PID=$!

# arm cleanup
trap "kill -9 $SERVER_PID $CLIENT_PID &> /dev/null" EXIT

# wait for client and server to complete
wait $CLIENT_PID; CLIENT_RC=$?
wait $SERVER_PID; SERVER_RC=$?

pass_fail_by_code client $CLIENT_RC server $SERVER_RC && pass_fail $CLIENT_LOG $TESTID
