#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
# run each test
$DIR/test0.sh
$DIR/test1.sh
$DIR/test2.sh

