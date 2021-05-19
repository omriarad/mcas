# 
#    Copyright [2021] [IBM Corporation]
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#        http://www.apache.org/licenses/LICENSE-2.0
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#

import pymmcore
import pymm
import numpy as np
import gc
import sys
import copy

from .memoryresource import MemoryResource
from .check import methodcheck

# common functions for shelved types
#
class ShelvedCommon:
    '''
    Common superclass for shelved objects
    '''
    def value_only_transaction(self, F, *args):
        self._value_named_memory.tx_begin()
        result = F(*args)
        self._value_named_memory.tx_commit()
        return result

    def __getattr__(self, name):
        if name == 'memory':
            return (self._value_named_memory.addr(), self._metadata_named_memory.addr())
        else:
            raise AttributeError()
            

class Shadow:
    '''
    Indicate type is a shadow type
    '''
    pass

class shelf():
    '''
    A shelf is a logical collection of variables held in CXL or persistent memory
    '''
    def __init__(self, name, size_mb=32):
        self.name = name
        self.mr = MemoryResource(name, size_mb)
        # todo iterate data and check value-metadata pairs
        print(self.mr)

    def __setattr__(self, name, value):
        # prevent implicit replacement (at least for the moment)
        print('setattr: ', name)
#        if name in self.__dict__:
#            if name == 'name' or name == 'mr':
#                raise RuntimeError('cannot change shelf attribute')
#            raise RuntimeError('cannot implicity replace object. use erase first')

        # check for supported types
        if isinstance(value, pymm.ndarray):
            self.__dict__[name] = value.make_instance(self.mr, name)
            return
        elif issubclass(type(value), pymm.ShelvedCommon):
            pass
        elif name == 'name' or name == 'mr': # allow our __init__ assignments
            self.__dict__[name] = value
        else:
            raise RuntimeError('cannot create this type (' + str(type(value)) + ') of object on the shelf')

        
    @methodcheck(types=[str])
    def erase(self, name):
        '''
        Erase and remove variable from the shelf
        '''
        # check the thing we are trying to erase is on the shelf
        if not name in self.__dict__:
            raise RuntimeError('attempting to erase something that is not on the shelf')

        # sanity check
        if sys.getrefcount(self.__dict__[name]) != 2:
            raise RuntimeError('erase failed due to outstanding references')

        self.__dict__.pop(name)
        gc.collect() # force gc
        
        # then remove the named memory from the store
        self.mr.erase_named_memory(name)



# NUMPY
#
# Note: all invocations define a transaction boundary
#-----------------------------------------------------
# import numpy as np
# import pyarrow as pa
#
# myShelf = pymm.shelf("/mnt/pmem0/data.pm")
#
# >> create an array in persistent memory
#
# myShelf.x = pymm.ndarray((9,9),dtype=np.uint8)
#
# >> create reference to array in persistent memoty
#
# X = myShelf.x
#
# >> in place modification (zero-copy)
#
# X.fill(8)
# X[1,1] = 99
#
# >> copy to DRAM
#
# c = X.copy()
#
# >> replication in persistent memory
#
# myShelf.z = X   or myshelf.z = myshelf.x
#
# >> remove from persistent memory (requires no outstanding references)
#
# myShelf.erase('z')




# APACHE ARROW
#------------------------------------------------
# import numpy as np
# import pyarrow as pa
#
# myShelf = pymm.shelf("/mnt/pmem0/data.pm")
#
# >> create an array of strings builder
#
# myShelf.x = pymm.arrow.StringBuilder()
#
# X = myShelf.x
#
# X.append("hello")
# X.append("world");
#
# X.finish();
#
# >> access immutable array (StringScalar type)
#
# print(X[0]) --> <pyarrow.StringScalar: 'hello'>
#
# >> create a schema ...
#

