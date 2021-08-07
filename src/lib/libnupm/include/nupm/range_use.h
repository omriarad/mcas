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

#ifndef MCAS_NUPM_RANGE_USE_H_
#define MCAS_NUPM_RANGE_USE_H_

#include <common/byte_span.h>
#include <common/memory_mapped.h>
#include <common/moveable_ptr.h>
#include <gsl/pointers>
#include <vector>

namespace nupm
{
struct range_manager;

struct range_use
{
private:
  common::moveable_ptr<range_manager> _rm;
  /* Note: arena_fs used multiple ranges */
  std::vector<common::memory_mapped> _iovm;

  std::vector<common::memory_mapped> address_coverage_check(std::vector<common::memory_mapped> &&iovm);
  using byte = char;
  using byte_span = common::byte_span;
public:
  byte_span operator[](std::size_t i) const { const auto &iov = _iovm.at(i).iov(); return common::make_byte_span(::base(iov), ::size(iov)); }
  range_use(range_manager *rm, std::vector<common::memory_mapped> &&);
  range_use(const range_use &) = delete;
  range_use &operator=(const range_use &) = delete;
  range_use(range_use &&) noexcept = default;
  void grow(std::vector<common::memory_mapped> &&);
  void shrink(std::size_t size);
  ~range_use();
  gsl::not_null<range_manager *> rm() const { return _rm; }
  ::off_t size() const;
};
}
#endif

