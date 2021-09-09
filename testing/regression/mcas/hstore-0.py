#!/usr/bin/python3

from configs import config_0
from shard_protos import shard_proto_dax
import argparse
import dax
import json
import re
import stores
import subprocess

class hstore_0(config_0):
	def __init__(self, hstoretype, dax_prefix, ipaddr, port=None, mm_plugin_path=None, count=1, accession=0):
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
		config_0.__init__(self, shard_proto_dax(ipaddr, h, dax_ctor(pfx=dax_prefix, accession=accession)), count)

from sys import argv

parser = argparse.ArgumentParser(description='Generate a JSON document for hstore testing.')
parser.add_argument("store", help="store type e.g. hstore", choices=["hstore", "hstore-cc", "hstore-mm", "hstore-mt", "hstore-cc-pe", "hstore-mm-pe"])
parser.add_argument("dax_prefix", help="DAX prefix (e.g. /dev/dax0, /mnt/pmem1)")
# As choices for ipaddr, use the IP address from the IP config
ip=subprocess.check_output(["/usr/sbin/ip", "-4", "-br", "-j", "addr"])
ij=json.loads(ip)
parser.add_argument("ipaddr", help="IP address of the port to use for the server", choices=[addr['local'] for ifc in map(lambda x: x['addr_info'], ij) for addr in ifc])
parser.add_argument("port", type=int, nargs='?', help="IP port number of the port to use for the server")
parser.add_argument('--mm-plugin-path', help='storage plugin path')
parser.add_argument("--shard-count", type=int, default=1, nargs='?', help="number of shards to configure")
parser.add_argument("--region", type=int, default=0, help="region within numa node to use")

args = parser.parse_args()

print(hstore_0(args.store, args.dax_prefix, args.ipaddr, args.port, args.mm_plugin_path, args.shard_count, args.region).json())
