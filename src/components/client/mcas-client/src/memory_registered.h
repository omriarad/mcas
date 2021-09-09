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

#ifndef MCAS_CLIENT_MEMORY_REGISTERED_H
#define MCAS_CLIENT_MEMORY_REGISTERED_H

#include "connection.h" /* mcas::client::Fabric_transport::buffer_base */
#if 0

#include "protocol.h"
#endif
#include "range.h"
#if 0
#include <common/cycles.h>
#include <unistd.h>
#endif
#include <common/delete_copy.h>
#include <common/moveable_ptr.h>
#include <common/perf/tm.h>
#include <api/mcas_itf.h> /* component::IMCAS::memory_handle_t */
#include <api/registrar_memory_direct.h>
#include <vector>

struct memory_registered
{
	using Registrar_memory_direct = component::Registrar_memory_direct;
	using IMCAS = component::IMCAS;
private:
	common::moveable_ptr<void> _desc;
	common::moveable_ptr<Registrar_memory_direct> _rmd;
	component::IMCAS::memory_handle_t _h;

public:
	memory_registered(TM_ACTUAL Registrar_memory_direct * rmd_,
                          const mcas::range<char *> &range_  // range to register
                          , void *desc_
                          )
		: _desc(desc_)
		, _rmd( desc_ ? nullptr : rmd_)
		, _h(_rmd ? _rmd->register_direct_memory(range_.first, range_.length()) : IMCAS::MEMORY_HANDLE_NONE)
	{
    TM_SCOPE()
	}

	memory_registered(Registrar_memory_direct * rmd_,
                          const mcas::range<char *> &range_  // range to register
                          , component::IKVStore::memory_handle_t handle_
                          )
		: _desc( handle_ == IMCAS::MEMORY_HANDLE_NONE ? nullptr : static_cast<mcas::client::Fabric_transport::buffer_base *>(_h)->get_desc() )
		, _rmd( _desc ? nullptr : rmd_ )
		, _h(_rmd ? _rmd->register_direct_memory(range_.first, range_.length()) : IMCAS::MEMORY_HANDLE_NONE)
	{
	}
  DELETE_COPY(memory_registered);
  memory_registered(memory_registered &&) noexcept = default;
  virtual ~memory_registered()
  {
    if (_rmd) {
      _rmd->unregister_direct_memory(_h);
    }
  }
  void *desc() const { return _h == IMCAS::MEMORY_HANDLE_NONE ? _desc.get() : static_cast<mcas::client::Fabric_transport::buffer_base *>(_h)->get_desc(); }
};

#endif
