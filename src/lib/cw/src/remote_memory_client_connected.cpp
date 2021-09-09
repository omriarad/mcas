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

using cw::remote_memory_client_connected;

remote_memory_client_connected::remote_memory_client_connected(
	std::shared_ptr<registered_memory> rm_out_
	, std::shared_ptr<registered_memory> rm_in_
	, cvk_type cnxn_and_vaddr_and_key_
	, quit_option quit_flag_
)
	: remote_memory_accessor()
	, _rm_out(rm_out_)
	, _rm_in(rm_in_)
	, _cnxn_and_vaddr_and_key(cnxn_and_vaddr_and_key_) /* get_rm(_cnxn, context_expected_, &(*_rm_out)[0])) */
	, _quit_flag(quit_flag_)
	, _last_stat(::E_FAIL)
{}

void remote_memory_client_connected::check_complete_static(void *t_, void *ctxt_, ::status_t stat_, std::size_t len_)
try
{
	/* The callback context must be the object which was polling. */
	(void)t_;
	assert(t_ == ctxt_);
	auto rmc = static_cast<remote_memory_client_connected *>(ctxt_);
	assert(rmc);
	rmc->check_complete(stat_, len_);
}
catch ( std::exception &e )
{
	std::cerr << "cw::remote_memory_client::" << __func__ << e.what() << "\n";
}

void remote_memory_client_connected::check_complete(::status_t stat_, std::size_t)
{
	_last_stat = stat_;
}

void remote_memory_client_connected::send_disconnect(component::IFabric_endpoint_comm &cnxn_, cw::registered_memory &rm_, cw::quit_option quit_flag_)
{
	send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
}

remote_memory_client_connected::~remote_memory_client_connected()
{
	try
	{
		if ( _quit_flag != cw::quit_option::remain )
		{
			send_disconnect(cnxn(), rm_out(), _quit_flag);
		}
	}
	catch ( std::exception &e )
	{
		std::cerr << __func__ << " exception " << e.what() << std::endl;
	}
}

status_t remote_memory_client_connected::wait_complete()
{
	cw::wait_poll(
		cnxn()
		, [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *)
			{
				check_complete_static(this, ctxt_, stat_, len_);
			}
	);
	if ( _last_stat != ::S_OK )
	{
		std::cerr << "cw::remote_memory_client::" << __func__ << ": " << _last_stat << "\n";
	}
	return _last_stat;
}

status_t remote_memory_client_connected::write(const common::string_view msg_)
{
	std::copy(msg_.begin(), msg_.end(), common::pointer_cast<char>(&rm_out()[0]));
	return write_uninitialized(msg_.size());
}

status_t remote_memory_client_connected::write_uninitialized(std::size_t sz_)
{
	::iovec buffers[1] = { ::iovec{ &rm_out()[0], sz_ } };
	void *desc[1] = { rm_out().desc() };
	cnxn().post_write(buffers, desc, vaddr(), key(), this);
	return wait_complete();
}

/* MCAS SRC -> TEST DEST: write to test area from provided source and descriptor (safe in MCAS before or after *actual* write) */
void *remote_memory_client_connected::write_to_test_remote_addr() const
{
	return reinterpret_cast<void *>(vaddr());
}

/* MCAS SRC -> TEST DEST: write to test area from provided source and descriptor (safe in MCAS before or after *actual* write) */
status_t remote_memory_client_connected::write_to_test(void *src_, std::size_t sz_, void *desc_)
{
	::iovec buffers[1] = { ::iovec{ src_, sz_ } };
	void *desc[1] = { desc_ };
	cnxn().post_write(buffers, desc, vaddr(), key(), this);
	return wait_complete();
}

void * remote_memory_client_connected::write_from_test_local_addr() const
{
	return &rm_out()[0];
}

/* TEST buffer <- MCAS remote SRC: read to test data buffer from remote target (safe in MCAS if followed by *actual* write */
status_t remote_memory_client_connected::read_to_test(std::size_t sz_, std::uint64_t vaddr_, std::uint64_t key_ )
{
	::iovec buffers[1] = { ::iovec{ &rm_out()[0], sz_ } };
	void *desc[1] = { rm_out().desc() };
	cnxn().post_read(buffers, desc, vaddr_, key_, this);
	return wait_complete();
}

/* TEST SRC -> MCAS DEST: write from local test data to designated remote target (safe in MCAS if followed by *actual* write,
 * or of preceded by a read_to_test) */
status_t remote_memory_client_connected::write_from_test(std::size_t sz_, std::uint64_t vaddr_, std::uint64_t key_ )
{
	::iovec buffers[1] = { ::iovec{ &rm_out()[0], sz_ } };
	void *desc[1] = { rm_out().desc() };
	cnxn().post_write(buffers, desc, vaddr_, key_, this);
	return wait_complete();
}

status_t remote_memory_client_connected::read(std::size_t sz_)
{
	::iovec buffers[1] = { ::iovec{ &rm_in()[0], sz_ } };
	cnxn().post_read(buffers, vaddr(), key(), this);
	return wait_complete();
}

status_t remote_memory_client_connected::read_verify(const common::string_view msg_)
{
	auto st =  read(msg_.size());
	std::string remote_msg(common::pointer_cast<char>(&rm_in()[0]), msg_.size());
	assert(msg_ == remote_msg);
	return st;
}

std::size_t remote_memory_client_connected::max_message_size() const
{
	return cnxn().max_message_size();
}
