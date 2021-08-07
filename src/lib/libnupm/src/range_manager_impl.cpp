/*
   Copyright [2017-2021] [IBM Corporation]
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

#include "range_manager_impl.h"

#include <cstddef>
#include <memory>
#include <stdexcept>

using nupm::range_manager_impl;

range_manager_impl::range_manager_impl(
	const common::log_source &ls_
)
  : common::log_source(ls_)
  , _address_coverage(std::make_unique<byte_interval_set>())
  , _address_fs_available(std::make_unique<byte_interval_set>())
{
  /* Maximum expected need is about 6 TiB (12 512GiB DIMMs).
   * Start, arbitrarily, at 0x10000000000
   */
  byte *free_address_begin = reinterpret_cast<byte *>(uintptr_t(1) << 40);
  auto free_address_end    = free_address_begin + (std::size_t(1) << 40);
  auto i = boost::icl::interval<byte *>::right_open(free_address_begin, free_address_end);
  _address_fs_available->insert(i);
}

range_manager_impl::~range_manager_impl()
{
  CPLOG(0, "%s::%s", typeid(*this).name(), __func__);
}

void * range_manager_impl::locate_free_address_range(std::size_t size_) const
{
	for ( auto i : *_address_fs_available )
	{
		if ( ptrdiff_t(size_) <= i.upper() - i.lower() )
		{
			return i.lower();
		}
	}
	throw std::runtime_error(__func__ + std::string(" out of address ranges"));
}

bool range_manager_impl::interferes(byte_interval coverage_) const
{
	return intersects(*_address_coverage, coverage_);
}

void range_manager_impl::add_coverage(byte_interval_set coverage_)
{
	*_address_coverage += coverage_;
	*_address_fs_available -= coverage_;
}

void range_manager_impl::remove_coverage(byte_interval coverage_)
{
	_address_coverage->erase(coverage_);
	_address_fs_available->insert(coverage_);
}
