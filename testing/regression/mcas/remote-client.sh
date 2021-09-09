#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

NODE_IP="$(node_ip)"

typeset fi_log_level="Info"
typeset -i debug_level=1
# replace value of --cores with a value clamped at the actual number of cores
declare -a args
while [ 0 -ne ${#@} ]
do :
  # args[${#args[@]}]="$1"
  case $1 in
    --cores) :
      args[${#args[@]}]="$1"
      shift
      args[${#args[@]}]="$(clamp_cpu $1)"
    ;;
    --debug-level) :
      args[${#args[@]}]="$1"
      shift
      debug_level=${args[${#args[@]}]}
    ;;
    --fi-log-level) :
      shift
      fi_log_level=$1
    ;;
    *) :
      args[${#args[@]}]="$1"
    ;;
  esac
  shift
done

DEBUG=${DEBUG:-${debug_level}}
# launch client

# set MCAS_AUTH_ID to prevent many getuid syscalls
set -x
# [ 0 -lt $DEBUG ] && echo MCAS_AUTH_ID=42 FI_LOG_LEVEL="${fi_log_level}" strace -e !read,write -ff -tt $DIR/../bin/kvstore-perf --src_addr $NODE_IP ${args+"${args[@]}"}
# MCAS_AUTH_ID=42 FI_LOG_LEVEL="${fi_log_level}" strace -e !read,write -ff -tt $DIR/../bin/kvstore-perf --src_addr "$NODE_IP" ${args+"${args[@]}"}
[ 0 -lt $DEBUG ] && echo MCAS_AUTH_ID=42 FI_LOG_LEVEL="${fi_log_level}" $DIR/../bin/kvstore-perf --src_addr $NODE_IP ${args+"${args[@]}"}
MCAS_AUTH_ID=42 FI_LOG_LEVEL="${fi_log_level}" $DIR/../bin/kvstore-perf --src_addr "$NODE_IP" ${args+"${args[@]}"}
