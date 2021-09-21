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

#include <cstddef> /* size_t */

#include <api/components.h>
#include <api/fabric_itf.h> /* IFabric, IFabric_client, IFabric_server_factory, IFabric_server, Fabric_connection, memory_region_t, IFabric_endpoint_connected, fi_context2 */

#include <common/byte_span.h>
#include <common/delete_copy.h>
#include <common/logging.h>
#include <common/moveable_ptr.h>
#include <common/types.h>

#include <sys/uio.h> /* iovec */

#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <functional> /* function */
#include <vector>

using cw::remote_memory_accessor;

void remote_memory_accessor::send_memory_info(component::IFabric_endpoint_connected &cnxn_, registered_memory &rm_)
{
	std::uint64_t vaddr = reinterpret_cast<std::uint64_t>(&rm_[0]);
	std::uint64_t key = rm_.key();
	{
		FLOG("Server: memory addr {} key {:x}", reinterpret_cast<void*>(vaddr), key);
	}
	char msg[(sizeof vaddr) + (sizeof key)];
	std::memcpy(msg, &vaddr, sizeof vaddr);
	std::memcpy(&msg[sizeof vaddr], &key, sizeof key);
	send_msg(cnxn_, rm_, msg, sizeof msg);
}

void remote_memory_accessor::send_msg(component::IFabric_endpoint_connected &cnxn_, registered_memory &rm_, const void *msg_, std::size_t len_)
{	
	std::memcpy(&rm_[0], msg_, len_);
	std::vector<::iovec> v{{&rm_[0],len_}};
	std::vector<void *> d{rm_.desc()};
	try
	{
		cnxn_.post_send(v, &*d.begin(), this);
		wait_poll(
			cnxn_
			, [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t, void *) -> void
				{
					(void)ctxt_;
					(void)stat_;
					assert(ctxt_ == this);
					assert(stat_ == S_OK);
				}
		);
	}
	catch ( const std::exception &e )
	{
		FLOGM("exception {}", e.what());
	}
}
