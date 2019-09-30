#!/bin/bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:`pwd`/dist/lib
NODE_IP=$(ip addr | grep -Po 'inet \K10.0.0.[0-9]+')
DESC=basic-ado-test-0
TESTID=3

if [ -z "$DISPLAY" ]; then
    echo '$DISPLAY not set; xterm would fail'
    exit 1
fi

# launch MCAS server
DAX_RESET=1 ./dist/bin/mcas --conf ./dist/testing/testing-ado0.conf --debug 0 &> test$TESTID-server.log &
SERVER_PID=$!

# give time to start server
sleep 3

#launch client
./dist/bin/ado-test --port 11911 --server $NODE_IP &> test$TESTID-client.log &
CLIENT_PID=$!

# arm cleanup
trap "kill -9 $SERVER_PID $CLIENT_PID &> /dev/null" EXIT

# wait for client to complete
tail --pid=$CLIENT_PID -f /dev/null

result=$(cat test$TESTID-client.log | grep -Po 'test0: rc=\K[0-9]+')
if [ "$result" == "0" ]; then
    echo -e "Test $TESTID: \e[32mpassed\e[0m"
else
    echo -e "Test $TESTID: \e[31mfail\e[0m"
fi
