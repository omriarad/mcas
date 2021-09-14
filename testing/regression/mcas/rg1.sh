#!/bin/bash

set -e -u

ulimit -a > ulimit.log
env > env.log
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$DIR/functions.sh"

typeset fi_log_level="Debug" 
typeset fi_log_level="Info" 
typeset debug=${DEBUG:-1}
typeset -i size_begin=4
size_begin=20
typeset -i cores_begin=1
typeset -r client_host="sg-mcas109.sg.almaden.ibm.com"
cores_begin=5

op=put_direct
op=get_direct
for ct in 1
do for lg in 23
	do sz=$((1<<lg))
		# STORE_SIZE assumes that only 1000 of the elements will be needed (i.e, that exp_perf_direct is modified to use ony a few elements
		# STORE_SIZE=$((1000*(200+8+$sz)*25/10))
		DEBUG=${debug} \
        FI_LOG_LEVEL="${fi_log_level}" \
		FI_MR_CACHE_MAX_SIZE=0x10000000000 \
		PERF_OPTS="--latency_range_max 5 --report_dir $(pwd) --report_tag $op-$sz-$ct" \
		VALUE_LENGTH=$sz ELEMENT_COUNT=1 CLIENT_HOST="$client_host" CLIENT_CORE_BASE=1 CLIENT_CORE_STRIDE=5 SHARD_COUNT=$ct \
		${DIR}/multi-hstore-$op-0.sh
	done
done | tee rg-$op.log
