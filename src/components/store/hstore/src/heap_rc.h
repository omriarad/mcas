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


#ifndef MCAS_HSTORE_HEAP_RC_H
#define MCAS_HSTORE_HEAP_RC_H

#include "heap.h"

#include "hstore_config.h"
#include "histogram_log2.h"
#include "persister_nupm.h"
#include "rc_alloc_wrapper_lb.h"
#include "trace_flags.h"
#include "tracked_header.h"

#include <common/byte_span.h>
#include <common/exceptions.h> /* General_exception */
#include <common/string_view.h>
#include <nupm/region_descriptor.h>

#include <sys/uio.h> /* iovec */

#include <algorithm>
#include <array>
#include <cstddef> /* size_t, ptrdiff_t */
#include <memory> /* unique_ptr */
#include <vector>

namespace nupm
{
	struct dax_manager_abstract;
}

namespace impl
{
	struct allocation_state_combined;
}

struct heap_rc_ephemeral;

struct heap_rc
	: private heap
{
private:
	tracked_header _tracked_anchor;
	std::unique_ptr<heap_rc_ephemeral> _eph;

public:
	explicit heap_rc(
		unsigned debug_level
		, byte_span pool0_full
		, byte_span pool0_heap
		, unsigned numa_node
		, string_view id
		, string_view backing_file
	);
	explicit heap_rc(
		unsigned debug_level
		, const std::unique_ptr<nupm::dax_manager_abstract> &dax_manager
		, string_view id
		, string_view backing_file
		, const std::uint64_t uuid
		, const byte_span *iov_addl_first
		, const byte_span *iov_addl_last
	);
	/* allocation_state_combined offered, but not used */
	explicit heap_rc(
		const unsigned debug_level_
		, const std::unique_ptr<nupm::dax_manager_abstract> &dax_manager_
		, const string_view id_
		, const string_view backing_file_
		, impl::allocation_state_combined const *
		, const std::uint64_t uuid_
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
	)
		: heap_rc(debug_level_, dax_manager_, id_, backing_file_, uuid_, iov_addl_first_, iov_addl_last_)
	{
	}

	heap_rc(const heap_rc &) = delete;
	heap_rc &operator=(const heap_rc &) = delete;

	~heap_rc();

    static constexpr std::uint64_t magic_value() { return 0xc74892d72eed493a; }

	auto grow(
		const std::unique_ptr<nupm::dax_manager_abstract> & dax_manager
		, std::uint64_t uuid
		, std::size_t increment
	) -> std::size_t;

	void quiesce();

	void *alloc(std::size_t sz, std::size_t alignment);
	void *alloc_tracked(std::size_t sz, std::size_t alignment);

	void inject_allocation(const void * p, std::size_t sz);

	void free(void *p, std::size_t sz, std::size_t alignment);
	void free_tracked(void *p, std::size_t sz, std::size_t alignment);

	unsigned percent_used() const;

	bool is_reconstituted(const void * p) const;

	/* debug */
	unsigned numa_node() const
	{
		return _numa_node;
	}

    nupm::region_descriptor regions() const;
};

#endif
