#!/bin/bash
ulimit -a > ulimit.log
env > env.log
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
. "$DIR/functions.sh"
DELAY=8

FSDAX_DIR=$(find_fsdax)
DEVDAX_PFX=$(find_devdax)

prefix()
{
  sleep $DELAY
}

run_hstore() {
  typeset ado_prereq="$1"
  shift
  # run each test
  prefix
  GOAL=750 ELEMENT_COUNT=10000 VALUE_LENGTH=2000000 PERF_OPTS="--report_tag put-$2" $DIR/mcas-hstore-put-0.sh $1
  : prefix
  : GOAL=2000 ELEMENT_COUNT=10000 VALUE_LENGTH=2000000 PERF_OPTS="--report_tag get-$2" $DIR/mcas-hstore-get-0.sh $1
  prefix
  GOAL=1250 FORCE_DIRECT=1 ELEMENT_COUNT=10000 VALUE_LENGTH=2000000 PERF_OPTS="--report_tag pd-$2" $DIR/mcas-hstore-put_direct-0.sh $1
  : prefix
  : GOAL=1900 FORCE_DIRECT=1 ELEMENT_COUNT=10000 VALUE_LENGTH=2000000 PERF_OPTS="--report_tag pd-$2" $DIR/mcas-hstore-get_direct-0.sh $1
}

DEVDAX_PFX=$(find_devdax)
if test -n "$DEVDAX_PFX"
then :
  DAX_PREFIX="$DEVDAX_PFX" USE_ODP=0 run_hstore has_module_mcasmod release $1
fi
