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

#include "pingpong_buffer_state.h"

buffer_state::buffer_state(component::IFabric_memory_control &cnxn_, std::size_t buffer_size_, std::uint64_t remote_key_, std::size_t msg_size_)
  : _rm{cnxn_, buffer_size_, remote_key_}
  , v{{&_rm[0], msg_size_}}
  , d{_rm.desc()}
{}
