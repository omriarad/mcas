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

#ifndef MCAS_CLIENT_MR_MANY_H
#define MCAS_CLIENT_MR_MANY_H

#include "memory_registered.h"
#include "range.h"
#include <common/perf/tm.h>

#include <api/registrar_memory_direct.h>
#include <vector>

struct mr_many
{
	using Registrar_memory_direct = component::Registrar_memory_direct;
	std::vector<memory_registered> _vec;
	mr_many(TM_ACTUAL Registrar_memory_direct *rmd_, gsl::span<const mcas::range<char *>> range_, gsl::span<const component::IKVStore::memory_handle_t> handles_)
		: _vec()
	{
		TM_SCOPE();
		_vec.reserve(std::size(range_));
		for ( std::size_t i = 0; i != std::size(range_); ++i )
		{
			_vec.emplace_back(
				TM_REF
				rmd_
				, range_[i]
				/* If no handle provided, or handle value is "HANDLE_NONE",
				 * Then ask memory_registered to create/destruct a one-time handle
				 * Else, use the provided handle.
				 */
				, handles_.size() <= i || handles_[i] == component::IKVStore::HANDLE_NONE ? nullptr : static_cast<mcas::client::Fabric_transport::buffer_base *>(handles_[i])->get_desc()
			);
		}
	}
	const memory_registered &at(std::size_t i) const { return _vec.at(i); }
};

#endif
