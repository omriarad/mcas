/*
   Copyright [2017-2019] [IBM Corporation]
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#include "rc_alloc_lb.h"

#include "avl_malloc.h"
#include "slab.h"
#include "region.h"
#include <common/exceptions.h>
#include <common/logging.h>
#include <common/utils.h>
#include <numa.h>
#include <stdexcept>

#define RCA_LB_PREFIX "Rca_LB:"

namespace nupm
{
Rca_LB::Rca_LB(unsigned debug_level_) : _rmap(new Region_map(debug_level_)) {}

Rca_LB::~Rca_LB() {}

void Rca_LB::add_managed_region(void * region_base,
                                size_t region_length,
                                int    numa_node)
{
  if (!region_base || region_length == 0 || numa_node < 0)
    throw std::invalid_argument("add_managed_region");

  _rmap->add_arena(region_base, region_length, numa_node);

  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d add_managed_region(%p,%lu,%d)", __FILE__, __LINE__, region_base, region_length, numa_node);
    debug_dump();
  }
}

void Rca_LB::inject_allocation(void *ptr, size_t size, int numa_node)
{
  _rmap->inject_allocation(ptr, size, numa_node);
  
  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d inject_allocation(ptr=%p, size=%lu, numanode=%d",
         __FILE__, __LINE__, ptr, size, numa_node);
    debug_dump();
  }  
}

void *Rca_LB::alloc(size_t size, int numa_node, size_t alignment)
{
  if (size == 0 || numa_node < 0)
    throw std::invalid_argument("Rca_LB::alloc invalid size or numa node");

  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d alloc(size=%lu, numanode=%d, alignment=%lu)",
         __FILE__, __LINE__, size, numa_node, alignment);
  }

  void * result = _rmap->allocate(size, numa_node, alignment);
  if(result == nullptr) {
    PWRN("Region allocator unable to allocate (size=%lu, alignment=%lu)", size, alignment);
    throw std::bad_alloc();
  }

  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d alloc(size=%lu, numanode=%d, alignment=%lu) (result=%p,size=%lu)",
         __FILE__, __LINE__, size, numa_node, alignment, result, size);
    debug_dump();
  }

  return result;
}

void Rca_LB::free(void *ptr, int numa_node, size_t size)
{
  if (numa_node < 0) throw std::invalid_argument("invalid numa_node");
  if (!ptr) throw std::invalid_argument("ptr argument is null");

  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d free(ptr=%p, numanode=%d, size=%lu)",
         __FILE__, __LINE__, ptr, numa_node, size);
  }

  _rmap->free(ptr, numa_node, size);

  if(_debug_level > 2) {
    PLOG(RCA_LB_PREFIX "%s:%d", __FILE__, __LINE__);
    debug_dump();
  }
}

void Rca_LB::debug_dump(std::string *out_log)
{
  if(_debug_level > 0) {
    std::cout << "----- Rca_LB -----------------\n";
    _rmap->arena_allocator().debug_dump(out_log);
    std::cout << "------------------------------\n";
  }
}

}  // namespace nupm
