#!/bin/bash
set -e
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

typeset -r DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

typeset -r SHARD_COUNT=${SHARD_COUNT:-2}
typeset -r PORT_BASE=${PORT_BASE:-11911}
typeset -r DAXTYPE="${DAXTYPE:-$(choose_dax_type)}"
# testname-keylength-valuelength-store-netprovider
typeset -r TESTID="mcas-$STORE-$PERFTEST-$KEY_LENGTH-$VALUE_LENGTH-$DAXTYPE-$SHARD_COUNT"
typeset -r CLIENT_HOST=${CLIENT_HOST:-$(hostname -f)}
typeset -r CLIENT_CORE_BASE=${CLIENT_CORE_BASE:-14}
typeset -r CLIENT_CORE_STRIDE=${CLIENT_CORE_STRIDE:-5}

# parameters for MCAS server and client
typeset -r NODE_IP=${NODE_IP:-"$(node_ip)"}
typeset -r DEBUG=${DEBUG:-0}
typeset -r PERF_OPTS=${PERF_OPTS:-"--skip_json_reporting"}

typeset -r CONFIG_STR="$("$DIR/hstore-0.py" "$STORE" "$DAXTYPE" "$NODE_IP" "$PORT_BASE" "$SHARD_COUNT")"
# launch MCAS server
[ 0 -lt $DEBUG ] && echo DAX_RESET=1 ./dist/bin/mcas --config \'"$CONFIG_STR"\' --forced-exit --debug $DEBUG
DAX_RESET=1 ./dist/bin/mcas --config "$CONFIG_STR" --forced-exit --debug $DEBUG &> test$TESTID-server.log &
typeset -r SERVER_PID=$!

sleep 3

# launch client
client_log() {
  echo "test$TESTID-$1-client.log"
}

ELEMENT_COUNT=$(scale_by_transport $ELEMENT_COUNT)
typeset -r ELEMENT_COUNT=$(scale $ELEMENT_COUNT $SCALE)

typeset -a CLIENT_PID
for SH in $(seq 0 $((SHARD_COUNT-1)))
do :
  port=$((PORT_BASE+SH))
  core=$(clamp_cpu $((CLIENT_CORE_BASE+SH*CLIENT_CORE_STRIDE)))
  [ 0 -lt $DEBUG ] && echo $DIR/remote-client.sh --cores "$core" --server $NODE_IP --port $port \
                          --test $PERFTEST --component mcas --elements $ELEMENT_COUNT --size $STORE_SIZE ${PERF_OPTS} \
                          --key_length $KEY_LENGTH --value_length $VALUE_LENGTH --report_tag $SH --debug_level $DEBUG
  log=$(client_log $SH)
  ssh "$CLIENT_HOST" $DIR/remote-client.sh --cores "$core" --server $NODE_IP --port $port \
                          --test $PERFTEST --component mcas --elements $ELEMENT_COUNT --size $STORE_SIZE ${PERF_OPTS} \
 		  --key_length $KEY_LENGTH --value_length $VALUE_LENGTH --report_tag $SH --debug_level $DEBUG &> $(client_log $SH) &
  CLIENT_PID[$SH]=$!
done

# arm cleanup
trap "kill -9 $SERVER_PID ${CLIENT_PID[*]} &> /dev/null" EXIT

# wait for client to complete
typeset -a CLIENT_RC
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

typeset -r GOAL=$(scale $GOAL $SCALE)
TOTAL=0
for SH in $(seq 0 $((SHARD_COUNT-1)))
do :
	r=$(pass_fail_by_code client ${CLIENT_RC[$SH]} server $SERVER_RC && pass_by_iops "$(client_log $SH)" $TESTID-$SH $GOAL)
  echo $r
  n=$(echo "$r" | sed -e 's/^Test .*[(]//' -e 's/ of [0-9][0-9]* IOPS[)]$//')
  if [[ "$n" =~ ^[0-9][0-9]*$ ]]
  then TOTAL=$((TOTAL+n))
  fi
done
echo $TESTID $TOTAL
