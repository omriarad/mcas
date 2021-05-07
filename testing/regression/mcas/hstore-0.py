#!/usr/bin/python3

from configs import config_0
from shard_protos import shard_proto_dax
import stores
import dax

class hstore_0(config_0):
    def __init__(self, hstoretype, daxtype, ipaddr, port=None, count=1):
        store_ctor = getattr(stores, hstoretype.replace('-', '_')) # hstore or hstore-cc
        h = store_ctor()
        #if port is not None:
        #    h.merge({"port": port})
        dax_ctor = getattr(dax, daxtype) # devdax or fsdax
        config_0.__init__(self, shard_proto_dax(ipaddr, h, dax_ctor()), count)

from sys import argv

# If, as for testing dax region conflicts, the port number is specified
if len(argv) == 6:
    print(hstore_0(argv[1], argv[2], argv[3], int(argv[4]), int(argv[5])).json())
elif len(argv) == 5:
    print(hstore_0(argv[1], argv[2], argv[3], int(argv[4])).json())
elif len(argv) == 4:
    print(hstore_0(argv[1], argv[2], argv[3]).json())
else:
    print("Usage: %s: {hstore|hstore-cc} {devdax|fadax} ipaddress [port] [count]" % argv[0])
