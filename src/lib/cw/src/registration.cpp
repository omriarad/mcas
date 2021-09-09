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

#include <boost/io/ios_state.hpp>

#include <sys/uio.h> /* iovec */

#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <functional> /* function */
#include <iostream> /* cerr */
#include <vector>

using cw::registration;

registration::registration()
	: _cnxn(nullptr)
	, _region()
	, _key()
	, _desc()
{}

registration::registration(
	gsl::not_null<component::IFabric_memory_control *> cnxn_
	, common::const_byte_span span_
	, std::uint64_t key_
	, std::uint64_t flags_
)
	: _cnxn(cnxn_)
	, _region(_cnxn->register_memory((PLOG("%s: C register %p:%zx", __func__, base(span_), size(span_)), span_), key_, flags_))
	, _key(_cnxn->get_memory_remote_key(_region))
	, _desc(_cnxn->get_memory_descriptor(_region))
{}

registration::~registration()
{
	if ( _cnxn )
	{
		try
		{
			_cnxn->deregister_memory(_region);
		}
		catch ( std::exception &e )
		{
			std::cerr << __func__ << " exception " << e.what() << std::endl;
		}
	}
}
