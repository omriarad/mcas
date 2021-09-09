#!/bin/bash

set -e -u

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
server=$1; shift
count=$1; shift
size=$1; shift
fi_log_level=${1:-Trace}; shift

FI_LOG_LEVEL=$fi_log_level COUNT=$count SIZE=$size $DIR/fabric-testcw_client "$server" 2>&1
