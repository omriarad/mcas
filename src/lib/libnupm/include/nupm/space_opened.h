/*
   Copyright [2020-2021] [IBM Corporation]
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


/*
 * Authors:
 *
 */

#ifndef MCAS_NUPM_OPENED_SPACE_H_
#define MCAS_NUPM_OPENED_SPACE_H_

#include "range_use.h"

#include "range_use.h"

#include <common/byte_span.h>
#include <common/fd_locked.h>
#include <common/logging.h> /* log_source */
#include <common/memory_mapped.h>
#include <common/types.h> /* addr_t */
#include <cstddef>
#include <vector>

struct range_manager;

namespace nupm
{
struct space_opened : private common::log_source
{
private:
  using byte_span = common::byte_span;
  common::fd_locked _fd_locked;
  /* Note: arena_fs may someday use multiple ranges */
  range_use _range;

  /* owns the file mapping */
  std::vector<common::memory_mapped> map_dev(int fd, addr_t base_addr, bool map_locked);
  std::vector<common::memory_mapped> map_fs(int fd, const std::vector<byte_span> &mapping, ::off_t offset);
public:
  space_opened(const common::log_source &, range_manager * rm_, common::fd_locked && fd, addr_t base_addr, bool map_locked);
  space_opened(const common::log_source &, range_manager * rm_, common::fd_locked && fd, const std::vector<byte_span> &mapping);
  space_opened(space_opened &&) noexcept = default;
  void shrink(std::size_t size);
  void grow(std::vector<byte_span> && iovv);
  int fd() const { return _fd_locked.fd(); }
  range_use &range() { return _range; }
  const range_use &range() const { return _range; }
};
}
#endif
