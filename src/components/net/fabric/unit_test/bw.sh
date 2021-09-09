#!/bin/bash

set -e -u

dir="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. ${dir}/functions.sh
node_ip="$(node_ip)"

remote=${REMOTE:-sg-mcas107.sg.almaden.ibm.com}
count=${COUNT:-10000}
size=${SIZE:-$((1<<23))}
fi_log_level=Debug # finest
fi_log_level=Info # normal

FI_LOG_LEVEL=$fi_log_level ${dir}/fabric-testbw 2>&1 | sed -e 's/libfabric:[0-9][0-9]*:/libfabric:/' > bw-server.log &
server_pid=$!

sleep 2
ssh $remote ${dir}/bw-client.sh ${node_ip} $count $size $fi_log_level 2>&1 | sed -e 's/libfabric:[0-9][0-9]*:/libfabric:/' > bw-client.log

# arm cleanup
trap "kill -9 $server_pid &> /dev/null" EXIT
wait $server_pid
