#!/usr/bin/python3
#
# discover an available pmem directory
#
import os

# Find first available /mnt/pmem<n> directory
def first_pmem():
    for i in range(0, 2):
        root='/mnt/pmem%d' % (i,)
        if os.path.isdir(root):
            return root

if __name__ == '__main__':
    print("First available pmem is %s" % (first_pmem(),))
