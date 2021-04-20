#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

NODE_IP="$(node_ip)"
DEBUG=${DEBUG:-0}

# replace value of --cores with a value clamped at the actual number of cores
declare -a args
while [ 0 -ne ${#@} ]
do :
  args[${#args[@]}]="$1"
  case $1 in
    --cores) :
      shift
      args[${#args[@]}]="$(clamp_cpu $1)"
    ;;
  esac
  shift
done

# launch client

[ 0 -lt $DEBUG ] && echo $DIR/../bin/kvstore-perf --src_addr $NODE_IP ${args+"${args[@]}"}
$DIR/../bin/kvstore-perf --src_addr "$NODE_IP" ${args+"${args[@]}"}
