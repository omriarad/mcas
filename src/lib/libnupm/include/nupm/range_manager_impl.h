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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef MCAS_NUPM_RANGE_MANAGER_IMPL_H_
#define MCAS_NUPM_RANGE_MANAGER_IMPL_H_

#include "range_manager.h"

#include <common/logging.h>

#include <boost/icl/interval_set.hpp>

namespace nupm
{
	/**
	 * Tracks used and available address ranges for a dax_manager instance
	 */
	struct range_manager_impl
		: protected common::log_source
		, public range_manager
	{
		explicit range_manager_impl(const common::log_source &);
		virtual ~range_manager_impl();

		bool interferes(byte_interval coverage) const override;
		void add_coverage(byte_interval_set coverage) override;
		void remove_coverage(byte_interval coverage) override;
		void * locate_free_address_range(std::size_t size) const override;
		std::unique_ptr<byte_interval_set>        _address_coverage;
		std::unique_ptr<byte_interval_set>        _address_fs_available;
	};
}  // namespace nupm

#endif
