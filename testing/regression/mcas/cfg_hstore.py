#!/usr/bin/python3

from configs import config
from shard_protos import shard_proto_dax
from numa_cores import numa_cores
import dax
import re
import stores

class cfg_hstore(config):
    def __init__(self, ipaddr, hstoretype, dax_prefix, port=None, count=1, numa_node=0, mm_plugin_path=None, accession=0):
        cores=numa_cores(numa_node)
        store_ctor = getattr(stores, hstoretype.replace('-', '_')) # hstore, hstore-cc, hstore-mc, hstore-mr
        h = store_ctor()
        if port is not None:
            h.merge({"port": port})
        if mm_plugin_path is not None:
            h.merge({"mm_plugin_path": mm_plugin_path})
        daxtype = re.match("/mnt/",dax_prefix) and "fsdax" or re.match("/dev/dax",dax_prefix) and "devdax" or None
        if not daxtype:
            raise Exception("Could not determine DAX type from prefix %s" % dax_prefix)
        dax_ctor = getattr(dax, daxtype) # devdax or fsdax
        config.__init__(self, shard_proto_dax(ipaddr, h, dax_ctor(pfx=dax_prefix, accession=accession), cores=cores), count)

if __name__ == '__main__':
    from argparse_cfg_hstore import argparse_cfg_hstore
    parser = argparse_cfg_hstore()
    args = parser.parse_args()
    print(cfg_hstore(args.ipaddr, args.store, args.dax_prefix, port=port=args.port, count=args.shard_count, numa_node=args.numa_node, args.mm_plugin_path, args.region).json())
