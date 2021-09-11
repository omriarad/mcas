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

#ifndef MCAS_HSTORE_SESSION_H
#define MCAS_HSTORE_SESSION_H

#include "hstore_config.h"
#include "session_base.h"
#include <common/logging.h> /* log_source */

#if CW_TEST
#include <common/byte_span.h>
#endif

#include "alloc_key.h" /* AK_FORMAL */
#include "lock_result.h"
#include "persist_atomic_controller.h"
#include "construction_mode.h"

#include <common/string_view.h>
#include <common/time.h> /* tsc_time_t, epoch_time_t */
#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <tuple>
#include <string>
#include <functional>

template <typename Table, typename Allocator>
	struct definite_lock;

template <typename Iterator>
	struct pool_iterator;

namespace nupm
{
	struct dax_manager_abstract;
}

/* hstore::open_pool_type, hstore::alloc_type, hstore::table_type, iIKVStore::lock_type_t */
template <typename Handle, typename Allocator, typename Table, typename LockType>
	struct session
		: public session_base<Handle>
	{
	private:
		using base = session_base<Handle>;
		using handle_type = Handle;
		using typename base::pool_type;

		using allocator_type = Allocator;
		using table_type = Table;
		using lock_type = LockType;

		using key_type = typename table_type::key_type;
		using mapped_type = typename table_type::mapped_type;
		using data_type = typename std::tuple_element<0, mapped_type>::type;
		using pool_iterator_type = pool_iterator<typename table_type::const_iterator>;
		using definite_lock_type = definite_lock<table_type, allocator_type>;
#if CW_TEST
		using byte_span = common::byte_span;
#endif
		using string_view_byte = common::basic_string_view<common::byte>;
		using string_view_key = string_view_byte;
		using string_view_value = string_view_byte;

		allocator_type _heap;
		bool _is_crash_consistent;
		bool _pin_seq; /* used only for force undo_redo call */
		table_type _map;
		impl::persist_atomic_controller<table_type> _atomic_state;
		std::map<pool_iterator_type *, std::shared_ptr<pool_iterator_type>> _iterators;

		static bool try_lock(typename std::tuple_element<0, mapped_type>::type &d, lock_type type);

		auto allocator() const noexcept -> allocator_type { return _heap; }
		auto locate_map(string_view_key key) -> table_type &;
		auto locate_map(string_view_key key) const -> const table_type &;

		bool undo_redo_pin_data(
			AK_FORMAL0
		);

		bool undo_redo_pin_key(
			AK_FORMAL0
		);

	public:
		/* PMEMoid, persist_data_t */
		template <typename OID, typename Persist>
			explicit session(
				AK_FORMAL
				unsigned debug_level
				, OID heap_oid_
				, handle_type &&pop
				, Persist *persist_data
			);

		std::uint64_t writes() const;

		explicit session(
			AK_FORMAL
			unsigned debug_level
			, handle_type &&pop
			, construction_mode mode
		);

		~session();

#if CW_TEST
		byte_span scratchpad() const;
#endif

		session(const session &) = delete;
		session& operator=(const session &) = delete;
		/* session constructor and get_pool_regions only */
		/* handle_type is hstore::open_pool_t
		 *
		 * hstore::open_pool_type is hstore::pm::open_pool_handle;
		 * hstore::pm::open_pool_handle is hstore_nupm<Region, ...>::open_pool_handle
		 * hstore_nupm<...> is ::open_pool<non_owner<Region>>, which isa non_owner<Region>
		 *  The non_owner<Region>::get method returns a Region *
		 *   region is region<persist_data_type, heap_alloc_shared_type>
		 *
		 * */
		const handle_type &handle() const;
		pool_type *pool() const { return handle().get(); }

		auto insert(
			AK_FORMAL
			TM_FORMAL
			string_view_key key,
			string_view_value value
		) -> std::pair<typename table_type::iterator, bool>;

		void update_by_issue_41(
			AK_FORMAL
			TM_FORMAL
			string_view_key key,
			string_view_value value,
			void * /* old_value */,
			const std::size_t old_value_len
		);

		auto get(
			TM_FORMAL
			string_view_key key,
			void* buffer,
			std::size_t buffer_size
		) const -> std::size_t;

		auto get_alloc(
			string_view_key key
		) const -> std::tuple<void *, std::size_t>;

		auto get_value_len(
			string_view_key key
		) const -> std::size_t;

#if ENABLE_TIMESTAMPS
		auto get_write_epoch_time(
			string_view_key key
		) const -> std::size_t;
#endif

		auto pool_grow(
			const std::unique_ptr<nupm::dax_manager_abstract> &dax_mgr_
			, const std::size_t increment_
		) const -> std::size_t;

		void resize_mapped(
			AK_FORMAL
			TM_FORMAL
			string_view_key key
			, std::size_t new_mapped_len
			, std::size_t alignment
		);

		auto lock(
			AK_FORMAL
			TM_FORMAL
			string_view_key key
			, lock_type type
			, void *value
			, std::size_t value_len
			, std::size_t alignment
		) -> lock_result;

		auto unlock_indefinite(
			TM_ACTUAL
			component::IKVStore::key_t key
			, component::IKVStore::unlock_flags_t flags
		) -> status_t;

		bool get_auto_resize() const;

		void set_auto_resize(bool auto_resize);

		auto erase(
			TM_FORMAL
			string_view_key key
		) -> status_t;

		auto count() const -> std::size_t;

		auto bucket_count() const -> std::size_t;

		auto map(
			std::function
			<
				int(string_view_key key
				, string_view_value value)
			> function_
		) -> void;

		auto map(
			std::function
			<
				int(string_view_key key
				, string_view_value value
				, common::tsc_time_t timestamp
				)
			> function_
			, common::epoch_time_t t_begin
			, common::epoch_time_t t_end
		) -> status_t;

		template <typename IT> /* *IT shall be a const component::IKVStore::Operation *const */
			void atomic_update_inner(
				AK_FORMAL
				TM_FORMAL
				const string_view_key key
				, table_type &map
				, IT first
				, IT last
				, lock_state lock
		);

		template <typename IT> /* *IT shall be a const component::IKVStore::Operation * */
			void atomic_update(
				AK_FORMAL
				TM_FORMAL
				string_view_key key
				, IT first
				, IT last
			);

		template <typename IT> /* *IT shall be a const component::IKVStore::Operation * */
			void lock_and_atomic_update(
				AK_FORMAL
				TM_FORMAL
				string_view_key key
				, IT first
				, IT last
		);

		void *allocate_memory(
			AK_FORMAL
			std::size_t size
			, std::size_t alignment
		);

		void free_memory(
			const void* addr
			, size_t size
		);

		void flush_memory(
			const void* addr
			, size_t size
		);

		unsigned percent_used() const;

		auto swap_keys(
			AK_FORMAL
			TM_FORMAL
			string_view_key key0
			, string_view_key key1
		) -> status_t;

		auto open_iterator() -> component::IKVStore::pool_iterator_t;

		status_t deref_iterator(
			component::IKVStore::pool_iterator_t iter
			, const common::epoch_time_t t_begin
			, const common::epoch_time_t t_end
			, component::IKVStore::pool_reference_t & ref
			, bool& time_match
			, bool increment
		);

		status_t close_iterator(component::IKVStore::pool_iterator_t iter);
	};

#include "session_base.tcc"
#include "session.tcc"

#endif
