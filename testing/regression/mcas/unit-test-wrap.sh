#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib

DIR="$(cd "$( dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
. "$DIR/functions.sh"

DEBUG=${DEBUG:-0}
TESTID=$(basename "$1")-"${STORE}"
if [ -n "$MM_PLUGIN_PATH" ]
then TESTID=${TESTID}-$(basename "$MM_PLUGIN_PATH")
fi
UNIT_LOG="test-$TESTID.log"

[ 0 -lt "$DEBUG" ] && echo DAX_RESET=1 STORE=\'"$STORE"\' STORE_LOCATION=\'"$STORE_LOCATION"\' MM_PLUGIN_PATH=\'"$MM_PLUGIN_PATH"\' \'"$DIR"/"$1"\'
DAX_RESET=1 "$DIR/$1" &> ${UNIT_LOG}
TEST_RC=$?

# check result

pass_fail_by_code unit-test $TEST_RC && pass_fail ${UNIT_LOG} $TESTID
