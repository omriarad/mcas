/*
   Copyright [2017-2020] [IBM Corporation]
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
#ifndef __mcas_REGION_MANAGER_H__
#define __mcas_REGION_MANAGER_H__

#include <api/fabric_itf.h>
#include <common/byte_span.h>
#include <common/logging.h>
#include <common/string_view.h>
#include <common/utils.h>

#include <set>

#include "connection_handler.h"
#include "memory_registered.h"
#include "types.h"

namespace mcas
{
class Region_manager : private common::log_source {
  using const_byte_span = common::const_byte_span;
 public:
	Region_manager(unsigned debug_level_, gsl::not_null<Connection *> conn) : common::log_source(debug_level_), _conn(conn), _reg{}
	{
		CPLOG(0, "%s", __func__);
	}

  Region_manager(const Region_manager&) = delete;
  Region_manager& operator=(const Region_manager&) = delete;

	virtual ~Region_manager()
	{
 	   CPLOG(0, "%s", __func__);
	}

  /**
   * Register memory with network transport for direct IO.  Cache in map.
   *
   * @param target Pointer to start or region
   * @param target_len Region length in bytes
   *
   * @return Memory region handle
   */
  memory_region_t ondemand_register(const_byte_span target)
  {
    std::uint64_t memory_key = 0;
    /* transport will now take care of repeat registrations */
    auto it =
      _reg.emplace(
        std::piecewise_construct
        , std::forward_as_tuple(::base(target))
        , std::forward_as_tuple(debug_level(), _conn, target, memory_key, 0)
      );
    CPLOG(0, "%s %p 0x%zx (total %zu)", __func__, ::base(target), ::size(target), _reg.size());
    return it->second.mr();
  }

  /* A registration at the greatest address address <= base(target) provides the region
   * 
   * (Since we are really asking about a range of memory, the input parameter should
   * be a span, not a single address. But at the moment we have not established a mechanism
   * to detect and report spans which are not encompassed by a single memory region.)
   */
  memory_region_t get_memory_region(const void *base) const
  {
    auto it = _reg.lower_bound(base);
    return
      it->first == base ? it->second.mr() /* exact hit */
      : it == _reg.begin() ? memory_region_t(0) /* no registered memory includes base */
      : (--it)->second.mr() /* if any registered memory includes base, return the memory immediately preceding it */
      ;
  }

 private:
  Connection*                    _conn;
  std::multimap<const void *, memory_registered<Connection>> _reg;
};
}  // namespace mcas

#endif  // __mcas_REGION_MANAGER_H__
