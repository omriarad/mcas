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

#include <api/kvstore_itf.h>

#include <common/byte.h>
#include <common/string_view.h>
#include <cstddef> /* size_t */

struct lock_result
{
	using string_view_byte = common::basic_string_view<common::byte>;
	using string_view_key = string_view_byte;
	using string_view_value = string_view_byte;
	enum class e_state
	{
		extant, created, not_created, creation_failed
	} state;
	component::IKVStore::key_t lock_key;
	string_view_key key;
	string_view_value value;
};

#endif
