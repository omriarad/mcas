#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

SHARD_COUNT=${SHARD_COUNT:-2}
PORT_BASE=${PORT_BASE:-11911}
DAXTYPE="${DAXTYPE:-$(choose_dax_type)}"
# testname-keylength-valuelength-store-netprovider
TESTID="mcas-$STORE-$PERFTEST-$KEY_LENGTH-$VALUE_LENGTH-$DAXTYPE"
CLIENT_HOST=${CLIENT_HOST:-hostname}

# parameters for MCAS server and client
NODE_IP=${NODE_IP:-"$(node_ip)"}
DEBUG=${DEBUG:-0}
PERF_OPTS=${PERF_OPTS:-"--skip_json_reporting"}

CONFIG_STR="$("$DIR/hstore-0.py" "$STORE" "$DAXTYPE" "$NODE_IP" "$PORT_BASE" "$SHARD_COUNT")"
# launch MCAS server
[ 0 -lt $DEBUG ] && echo DAX_RESET=1 ./dist/bin/mcas --config \'"$CONFIG_STR"\' --forced-exit --debug $DEBUG
DAX_RESET=1 ./dist/bin/mcas --config "$CONFIG_STR" --forced-exit --debug $DEBUG &> test$TESTID-server.log &
SERVER_PID=$!

sleep 3

# launch client
CLIENT_LOG="test$TESTID-client.log"
ELEMENT_COUNT=$(scale_by_transport $ELEMENT_COUNT)
ELEMENT_COUNT=$(scale $ELEMENT_COUNT $SCALE)

typeset -a CLIENT_PID
for SH in $(seq 0 $((SHARD_COUNT-1)))
do :
  PORT=$((PORT_BASE+SH))
  CORE=$(clamp_cpu $((14+SH)))
  [ 0 -lt $DEBUG ] && echo $DIR/remote-client.sh --cores "$CORE" --server $NODE_IP --port $PORT \
                          --test $PERFTEST --component mcas --elements $ELEMENT_COUNT --size $STORE_SIZE ${PERF_OPTS} \
                          --key_length $KEY_LENGTH --value_length $VALUE_LENGTH --debug_level $DEBUG
  ssh "$CLIENT_HOST" $DIR/remote-client.sh --cores "$CORE" --server $NODE_IP --port $PORT \
                          --test $PERFTEST --component mcas --elements $ELEMENT_COUNT --size $STORE_SIZE ${PERF_OPTS} \
                          --key_length $KEY_LENGTH --value_length $VALUE_LENGTH --debug_level $DEBUG &> "$CLIENT_LOG-$SH" &
  CLIENT_PID[$SH]=$!
done

# arm cleanup
trap "kill -9 $SERVER_PID ${CLIENT_PID[*]} &> /dev/null" EXIT

# wait for client to complete
typeset -a CLIENT_RCA
CLIENT_RC_MAX=0
for SH in $(seq 0 $((SHARD_COUNT-1)))
do :
  wait ${CLIENT_PID[$SH]}
  CLIENT_RC[$SH]=$?
  if [ $CLIENT_RC_MAX -lt $? ]; then CLIENT_RC_MAX=$?; fi
done
[ 0 -ne $CLIENT_RC_MAX ] && kill $SERVER_PID
wait $SERVER_PID; SERVER_RC=$?

# check result

GOAL=$(scale $GOAL $SCALE)
for SH in $(seq 0 $((SHARD_COUNT-1)))
do :
  pass_fail_by_code client ${CLIENT_RC[$SH]} server $SERVER_RC && pass_by_iops "$CLIENT_LOG-$SH" $TESTID $GOAL
done
