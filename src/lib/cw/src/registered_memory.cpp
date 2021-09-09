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

#include <cw/cw_common.h>

#include <cassert>

using cw::registered_memory;

registered_memory::registered_memory(gsl::not_null<component::IFabric_memory_control *> cnxn_, std::unique_ptr<memory> &&m_, std::uint64_t remote_key_)
	: _memory(std::move(m_))
	, _registration(cnxn_, common::make_const_byte_span(*_memory), remote_key_, 0U)
{}

registered_memory::registered_memory()
	: _memory()
	, _registration()
{}

common::byte &registered_memory::at(std::size_t ix)
{
	assert(ix < _memory->size());
	return *(_memory->data() + ix);
}
