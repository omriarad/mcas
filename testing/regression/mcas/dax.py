#!/usr/bin/python3

from dm import dm

# Default locations for devdax and fsdax stores are: /dev/dax<n>.0, /mnt/pmem<n>/a0

class fsdax(dm):
    """ fsdax specification (within a shard) """
    def __init__(self, pfx, accession=0):
        self.pfx = pfx
        self.accession = accession
        dm.__init__(self, {"path": "%s/a%d" % (pfx,accession)})
    def step(self,n):
        return fsdax(self.pfx, self.accession+n)

class devdax(dm):
    """ devdax specification (within a shard) """
    def __init__(self, pfx, accession=0):
        self.pfx = pfx
        self.accession = accession
        dm.__init__(self, {"path": "%s.%d" % (pfx, accession)})
    def step(self,n):
        return devdax(self.pfx, self.accession+n)

if __name__ == '__main__':
    print("fsdax:", fsdax().json())
    print("devdax:", devdax().json())
