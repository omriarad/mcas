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

#include "heap_cc.h"

#include "as_pin.h"
#include "as_emplace.h"
#include "as_extend.h"
#include "clean_align.h"
#include "heap_cc_ephemeral.h"
#include <ccpm/cca.h>
#include <common/env.h>
#include <common/byte_span.h>
#include <common/pointer_cast.h>
#include <common/string_view.h>
#include <common/utils.h> /* round_up */
#include <nupm/dax_manager_abstract.h>
#include <algorithm>
#include <cassert>
#include <memory> /* make_unique */
#include <stdexcept> /* range_error */
#include <utility>

namespace
{
#if USE_CC_HEAP == 4
	bool leak_check = common::env_value("LEAK_CHECK", false);
#endif
}

namespace
{
	using byte_span = common::byte_span;
	using string_view = common::string_view;
	byte_span open_region(const std::unique_ptr<nupm::dax_manager_abstract> &dax_manager_, string_view id_, unsigned numa_node_)
	{
		auto file_and_iov = dax_manager_->open_region(id_, numa_node_);
		if ( file_and_iov.address_map().size() != 1 )
		{
			throw std::range_error("failed to re-open region " + std::string(id_));
		}
		return file_and_iov.address_map().front();
	}

	const ccpm::region_vector_t add_regions_full(
		ccpm::region_vector_t &&rv_
		, const byte_span pool0_heap_
		, const std::unique_ptr<nupm::dax_manager_abstract> &dax_manager_
		, unsigned numa_node_
		, const uint64_t uuid_
		, const byte_span *iov_addl_first_
		, const byte_span *iov_addl_last_
		, std::uint64_t *first_, std::uint64_t *last_
	)
	{
		auto v = std::move(rv_);
		for ( auto it = first_; it != last_; ++it )
		{
			auto r = open_region(dax_manager_, std::to_string(uuid_ + *it), numa_node_);
			if ( it == first_ )
			{
				(void) pool0_heap_;
				VALGRIND_MAKE_MEM_DEFINED(::base(pool0_heap_), ::size(pool0_heap_));
				VALGRIND_CREATE_MEMPOOL(::base(pool0_heap_), 0, true);
				for ( auto a = iov_addl_first_; a != iov_addl_last_; ++a )
				{
					VALGRIND_MAKE_MEM_DEFINED(::base(*a), ::size(*a));
					VALGRIND_CREATE_MEMPOOL(::base(*a), 0, true);
				}
			}
			else
			{
				VALGRIND_MAKE_MEM_DEFINED(::base(r), ::size(r));
				VALGRIND_CREATE_MEMPOOL(::base(r), 0, true);
			}
			v.push_back(r);
		}

		return v;
	}
}

/* When used with ADO, this space apparently needs a 2MiB alignment.
 * 4 KiB produces sometimes produces a disagreement between server and AOo mappings
 * which manifest as incorrect key and data values as seen on the ADO side.
 */
heap_cc::heap_cc(
	const unsigned debug_level_
	, impl::allocation_state_emplace *const ase_
	, impl::allocation_state_pin *const aspd_
	, impl::allocation_state_pin *const aspk_
	, impl::allocation_state_extend *const asx_
	, const byte_span pool0_full_
	, const byte_span pool0_heap_
	, const unsigned numa_node_
	, const string_view id_
	, const string_view backing_file_

)
	: heap(pool0_full_, pool0_heap_, numa_node_)
	, _eph(
		std::make_unique<heap_cc_ephemeral>(
			debug_level_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, id_
			, backing_file_
			, std::vector<byte_span>(1, _pool0_full) // ccpm::region_vector_t(::base(_pool0_full), ::size(_pool0_heap))
			, pool0_heap_
		)
	)
{
	/* cursor now locates the best-aligned region */
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", ::base(_pool0_heap), " .. ", ::end(_pool0_heap)
		, " size ", ::size(_pool0_heap)
		, " new"
	);
	VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, false);
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winit-self"
#pragma GCC diagnostic ignored "-Wuninitialized"
heap_cc::heap_cc(
	const unsigned debug_level_
	, const std::unique_ptr<nupm::dax_manager_abstract> &dax_manager_
	, const string_view id_
	, const string_view backing_file_
	, const std::uint64_t uuid_
	, const byte_span *iov_addl_first_
	, const byte_span *iov_addl_last_
	, impl::allocation_state_emplace *const ase_
	, impl::allocation_state_pin *const aspd_
	, impl::allocation_state_pin *const aspk_
	, impl::allocation_state_extend *const asx_
)
	: heap(*this)
	, _eph(
		std::make_unique<heap_cc_ephemeral>(
			debug_level_
			, ase_
			, aspd_
			, aspk_
			, asx_
			, id_
			, backing_file_
			, add_regions_full(
				ccpm::region_vector_t(
					::data(_pool0_full), ::size(_pool0_full)
				)
				, _pool0_heap
				, dax_manager_
				, _numa_node
				, uuid_
				, iov_addl_first_
				, iov_addl_last_
				, &_more_region_uuids[0]
				, &_more_region_uuids[_more_region_uuids_size]
			)
			, _pool0_heap
			, [ase_, aspd_, aspk_, asx_] (const void *p) -> bool {
				/* To answer whether the map or the allocator owns pointer p?
				 * Guessing that true means that the map owns p
				 */
				auto cp = const_cast<void *>(p);
				return ase_->is_in_use(cp) || aspd_->is_in_use(p) || aspk_->is_in_use(p) || asx_->is_in_use(p);
			}
		)
	)
{
	hop_hash_log<trace_heap_summary>::write(
		LOG_LOCATION
		, " pool ", ::base(_pool0_heap), " .. ", ::end(_pool0_heap)
		, " size ", ::size(_pool0_heap)
		, " reconstituting"
	);
	VALGRIND_MAKE_MEM_DEFINED(::base(_pool0_heap), ::size(_pool0_heap));
	VALGRIND_CREATE_MEMPOOL(::base(_pool0_heap), 0, true);
}
#pragma GCC diagnostic pop

heap_cc::~heap_cc()
{
	quiesce();
}

auto heap_cc::regions() const -> nupm::region_descriptor
{
	return _eph->get_primary_region();
}

auto heap_cc::grow(
	const std::unique_ptr<nupm::dax_manager_abstract> & dax_manager_
	, std::uint64_t uuid_
	, std::size_t increment_
) -> std::size_t
{
	return heap::grow(_eph.get(), dax_manager_, uuid_, increment_);
}

void heap_cc::quiesce()
{
	hop_hash_log<trace_heap_summary>::write(LOG_LOCATION, " size ", ::size(_pool0_heap), " allocated ", _eph->_allocated);
	VALGRIND_DESTROY_MEMPOOL(::base(_pool0_heap));
	VALGRIND_MAKE_MEM_UNDEFINED(::base(_pool0_heap), ::size(_pool0_heap));
	_eph->write_hist<trace_heap_summary>(_pool0_heap);
	_eph.reset(nullptr);
}

void heap_cc::alloc(persistent_t<void *> *p_, std::size_t sz_, std::size_t align_)
{
	auto align = clean_align(align_, sizeof(void *));

	/* allocation must be multiple of alignment */
	auto sz = (sz_ + align - 1U)/align * align;

	try {
#if USE_CC_HEAP == 4
		if ( _eph->_aspd->is_armed() )
		{
		}
		else if ( _eph->_aspk->is_armed() )
		{
		}
		/* Note: order of testing is important. An extend arm+allocate) can occur while
		 * emplace is armed, but not vice-versa
		 */
		else if ( _eph->_asx->is_armed() )
		{
			_eph->_asx->record_allocation(&persistent_ref(*p_), persister_nupm());
		}
		else if ( _eph->_ase->is_armed() )
		{
			_eph->_ase->record_allocation(&persistent_ref(*p_), persister_nupm());
		}
		else
		{
			if ( leak_check )
			{
				PLOG(PREFIX "leaky allocation, size %zu", LOCATION, sz_);
			}
		}
#endif
		/* IHeap interface does not support abstract pointers. Cast to regular pointer */
		_eph->_heap->allocate(*reinterpret_cast<void **>(p_), sz, align);
		/* We would like to carry the persistent_t through to the crash-conssitent allocator,
		 * but for now just assume that the allocator has modifed p_, and call tick to indicate that.
		 */
		perishable::tick();

		VALGRIND_MEMPOOL_ALLOC(::base(_pool0_heap), p_, sz);
		/* size grows twice: once for aligment, and possibly once more in allocation */
		hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz_, "->", sz);
		_eph->_allocated += sz;
		_eph->_hist_alloc.enter(sz);
	}
	catch ( const std::bad_alloc & )
	{
		_eph->write_hist<true>(_pool0_heap);
		/* Sometimes lack of space will cause heap to throw a bad_alloc. */
		throw;
	}
}

void heap_cc::free(persistent_t<void *> *p_, std::size_t sz_)
{
	VALGRIND_MEMPOOL_FREE(::base(_pool0_heap), p_);
	auto sz = _eph->free(p_, sz_);
	hop_hash_log<trace_heap>::write(LOG_LOCATION, "pool ", ::base(_pool0_heap), " addr ", p_, " size ", sz_, "->", sz);
}

unsigned heap_cc::percent_used() const
{
	return
		unsigned(
			_eph->_capacity
			? _eph->_allocated * 100U / _eph->_capacity
			: 100U
		);
}

void heap_cc::extend_arm() const
{
	_eph->_asx->arm(persister_nupm());
}

void heap_cc::extend_disarm() const
{
	_eph->_asx->disarm(persister_nupm());
}

void heap_cc::emplace_arm() const { _eph->_ase->arm(persister_nupm()); }
void heap_cc::emplace_disarm() const { _eph->_ase->disarm(persister_nupm()); }

impl::allocation_state_pin &heap_cc::aspd() const { return *_eph->_aspd; }
impl::allocation_state_pin &heap_cc::aspk() const { return *_eph->_aspk; }

void heap_cc::pin_data_arm(
	cptr &cptr_
) const
{
#if USE_CC_HEAP == 4
	_eph->_aspd->arm(cptr_, persister_nupm());
#else
	(void)cptr_;
#endif
}

void heap_cc::pin_key_arm(
	cptr &cptr_
) const
{
#if USE_CC_HEAP == 4
	_eph->_aspk->arm(cptr_, persister_nupm());
#else
	(void)cptr_;
#endif
}

char *heap_cc::pin_data_get_cptr() const
{
#if USE_CC_HEAP == 4
	assert(_eph->_aspd->is_armed());
	return _eph->_aspd->get_cptr();
#else
	return nullptr;
#endif
}
char *heap_cc::pin_key_get_cptr() const
{
#if USE_CC_HEAP == 4
	assert(_eph->_aspk->is_armed());
	return _eph->_aspk->get_cptr();
#else
	return nullptr;
#endif
}

void heap_cc::pin_data_disarm() const
{
	_eph->_aspd->disarm(persister_nupm());
}

void heap_cc::pin_key_disarm() const
{
	_eph->_aspk->disarm(persister_nupm());
}
