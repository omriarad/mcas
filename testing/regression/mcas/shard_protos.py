#!/usr/bin/python3

from dm import dm

class shard_proto_ado_mixin(dm):
    """ additions to a shard for ado testing """
    def __init__(self):
        dm.__init__(self, {
            "ado_plugins" : ["libcomponent-adoplugin-testing.so"],
        })

# Each shard_proto is a class derived from dm, to indicate that
# some "dict deep merges" may be needed.
# For multiple shards, may want to make shard_proto a generator.

class shard_proto(dm):
    """ shard prototype """
    def __init__(self, ipaddr, backend, core=0, port=11911):
        #
        # steppable itens
        #
        self.ipaddr = ipaddr
        self.core = core
        self.port = port
        self.backend = backend
        dm.__init__(self, {
            "core" : core,
            "addr" : ipaddr,
            "port" : port,
        })
        self.merge(backend)
    def step(self,n):
        return shard_proto(self.ipaddr, self.backend, self.core+n, self.port+n)

class shard_proto_net(dm):
    """ shard prototype with network device """
    def __init__(self, ipaddr, backend, net, core=0, port=11911):
        dm.__init__(self, {
            "core" : core,
            "addr" : ipaddr,
            "port" : port,
            "net"  : net
        })
        self.merge(backend)


class shard_proto_dax(shard_proto):
    """ shard prototype (for dax) """
    def __init__(self, ipaddr, backend, path, memaddr=0x9000000000, core=0, port=11911):
        shard_proto.__init__(self, ipaddr, backend, core, port)
        self.path = path
        self.memaddr = memaddr
        p = dm({ "addr": memaddr })
        p.merge(path)
        self.merge({"dax_config": [ p.value() ]})
    def step(self,n):
        return shard_proto_dax(self.ipaddr, self.backend, self.path.step(n), self.memaddr+n*0x1000000000, self.core+n, self.port+n)

class shard_proto_ado(shard_proto):
    """ shard prototype (for ado) """
    def __init__(self, ipaddr, backend):
        shard_proto.__init__(self, ipaddr, backend)
        self.merge(shard_proto_ado_mixin())

class shard_proto_dax_ado(shard_proto_dax):
    """ shard prototype (for dax, ado) """
    def __init__(self, ipaddr, backend, path):
        shard_proto_dax.__init__(self, ipaddr, backend, path)
        self.merge(shard_proto_ado_mixin())
