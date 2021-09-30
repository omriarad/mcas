#!/usr/bin/python3 -m unittest
#
# test case which knows a usable pmem directory
#
import unittest
import pmem_discovery

main = unittest.main

class TestCase(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self,*args,**kwargs)

        self.pmem_root=pmem_discovery.first_pmem()
        self.assertNotEqual(self.pmem_root, '')

    #def setUp(self):
    #    pass

    def test_setup(self):

        print("first available pmem_path is %s" % (self.pmem_root,))

if __name__ == '__main__':
    unittest.main()
