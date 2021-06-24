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

#ifndef MCAS_HSTORE_LOCK_RESULT_H
#define MCAS_HSTORE_LOCK_RESULT_H

#include <api/kvstore_itf.h> // component::IKVStore

#include "lock_impl.h"
#include <common/byte.h>
#include <common/string_view.h>
#include <common/utils.h>
#include <cstddef> /* size_t */

/* Does lock complain if the locked value is misaligned? According to mapstore, no. */
#define RESULT_CHECKS_ALIGNMENT 0

struct lock_result
{
	using string_view_byte = common::basic_string_view<common::byte>;
	using string_view_key = string_view_byte;
	using string_view_value = string_view_byte;
	enum class e_state
	{
		extant, created, not_created, creation_failed,
#if RESULT_CHECKS_ALIGNMENT
		 misaligned,
#endif
	} state;
	component::IKVStore::key_t lock_key;
	string_view_value value;
	string_view_key key;
private:
	template <typename D, typename L>
		bool try_lock(D &d, L lock_type_)
		{
			return
				lock_type_ == component::IKVStore::STORE_LOCK_READ
				? d.try_lock_shared()
				: d.try_lock_exclusive()
				;
		}
public:
	/* failure cases only (not_created, creation_failed, early misaligned) */
	explicit lock_result(e_state s_, string_view_value value_)
		: state(s_)
		, lock_key(component::IKVStore::KEY_NONE)
		, value(value_)
		, key()
	{
	}

	/* good cases only (extant, created) */
	template <typename D, typename L>
		explicit lock_result(
			e_state s_
			, D &d_
			, L lock_type_
			, string_view_byte lock_key_
			, string_view_key key_
			, std::size_t
#if HSTORE_LOCK_RESULT_CHECKS_ALIGNMENT
				alignment_
#endif
		)
#if HSTORE_LOCK_RESULT_CHECKS_ALIGNMENT
			: state(
				check_aligned(d_.data_fixed(), alignment_) ? s_ : e_state::misaligned
			)
			, lock_key(
				s_ != e_state::misaligned && try_lock(d_, lock_type_)
				? new lock_impl(lock_key_)
				: component::IKVStore::KEY_NONE
			)
#else
			: state(s_)
			, lock_key(try_lock(d_, lock_type_) ? new lock_impl(lock_key_) : component::IKVStore::KEY_NONE)
#endif
			, value(d_.data_fixed(), d_.size())
			, key(key_)
		{
		}
};

#endif
