/*
   Copyright [2019-2020] [IBM Corporation]
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
#ifndef MCAS_REGISTERED_MEMORY_H
#define MCAS_REGISTERED_MEMORY_H

#include <common/byte_span.h>
#include <common/logging.h>
#include <common/moveable_ptr.h>

#include "mcas_config.h"
#include <cstdint> /* uint64_t, size_t */
#include <utility> /* swap */

namespace mcas
{
template <typename T>
struct memory_registered : private common::log_source {
private:
  using const_byte_span = common::const_byte_span;
  common::moveable_ptr<T>     _memory_control;
  typename T::memory_region_t _r;
  std::uint64_t _key;
  void * _desc;

 public:
  explicit memory_registered(unsigned      debug_level_,
                             T *           transport_,
                             const_byte_span region_,
                             std::uint64_t key_,
                             std::uint64_t flags_)
      : common::log_source(debug_level_)
      , _memory_control(transport_)
      , _r(_memory_control->register_memory(region_, key_, flags_))
      , _key(_memory_control->get_memory_remote_key(_r))
      , _desc(_memory_control->get_memory_descriptor(_r))
  {
    CPLOG(0, "%s %p (%p:0x%zx)", __func__, common::p_fmt(_r), ::base(region_), ::size(region_));
  }

  explicit memory_registered(unsigned      debug_level_,
                             T *           memory_control_,
		typename T::memory_region_t r_)
      : common::log_source(debug_level_)
      , _memory_control(nullptr)
      , _r(r_)
      , _key(memory_control_->get_memory_remote_key(_r))
      , _desc(memory_control_->get_memory_descriptor(_r))
  {
    CPLOG(0, "%s %p", __func__, common::p_fmt(_r));
  }

  memory_registered(const memory_registered &) = delete;
  memory_registered(memory_registered &&other_) noexcept = default;
  memory_registered &operator=(const memory_registered) = delete;

  ~memory_registered()
  {
    if (_memory_control) {
      CPLOG(2, "%s %p", __func__, common::p_fmt(_r));
      _memory_control->deregister_memory(_r);
    }
  }
  auto mr() const { return _r; }
  T *memory_control() const { return _memory_control; }
  auto desc() const { return _desc; }
  auto key() const { return _key; }
  auto get_memory_descriptor() const { return desc(); }
};

template <typename T>
  bool operator<(const memory_registered<T> &a, const memory_registered<T> &b)
  {
    return a.memory_control() < b.memory_control() || ( a.memory_control() == b.memory_control() && a.mr() < b.mr() );
  }

template <typename T>
memory_registered<T> make_memory_registered(unsigned      debug_level_,
                                            T *           transport_,
                                            void *        base_,
                                            std::size_t   len_,
                                            std::uint64_t key_,
                                            std::uint64_t flags_)
{
  return memory_registered<T>(debug_level_, transport_, base_, len_, key_, flags_);
}
}  // namespace mcas

#endif
