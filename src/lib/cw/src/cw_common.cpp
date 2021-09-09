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

/*
 * returns: number of polls (including the successful poll)
 */
unsigned cw::wait_poll(
	component::IFabric_endpoint_connected &comm_
	, std::function<void(
		void *context
		, ::status_t
		, std::uint64_t completion_flags
		, std::size_t len
		, void *error_data
	)> cb_
)
{
	std::size_t ct = 0;
	unsigned poll_count = 0;
	while ( ct == 0 )
	{
		++poll_count;
		ct += comm_.poll_completions(cb_);
	}
	/* poll_completions does not always get a completion after wait_for_next_completion returns
	 * (does it perhaps return when a message begins to appear in the completion queue?)
	 * but it should not take more than two trips through the loop to get the completion.
	 */
	assert(ct == 1);
	return poll_count;
}

auto cw::get_rm(
	component::IFabric_client *cnxn_
	, fi_context2 *expected_ctxt_
	, common::byte *rm_
) -> cvk_type
{
	cvk_type r{cnxn_, 0, 0};
	wait_poll(
		*cnxn_
		, [&r, expected_ctxt_, rm_] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
			{
				(void)ctxt_;
				(void)stat_;
				(void)len_;
				assert(ctxt_ == expected_ctxt_);
				assert(stat_ == ::S_OK);
				auto &vaddr = std::get<1>(r);
				auto &key = std::get<2>(r);
				assert(len_ == (sizeof vaddr) + (sizeof key));
				std::memcpy(&vaddr, rm_, sizeof vaddr);
				std::memcpy(&key, rm_ + (sizeof vaddr), sizeof key);
			}
	);
	boost::io::ios_base_all_saver sv(std::cerr);
	std::cerr << "Client: remote memory addr " << reinterpret_cast<void*>(std::get<1>(r)) << " key " << std::hex << std::get<2>(r) << std::endl;
	return r;
}
