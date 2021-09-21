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

#include <cw/cw_fabric_test.h>
#include <cw/cw_common.h>
#include <api/mcas_itf.h> /* cw::test_data */

#include <api/components.h>
#include <api/fabric_itf.h> /* IFabric, IFabric_client, IFabric_endpoint_comm, IFabric_endpoint_unconnected_client */

#include <common/env.h> /* env_value */
#include <common/errors.h> /* S_OK */
#include <common/histogram.h>
#include <common/logging.h>
#include <common/string_view.h>
#include <common/types.h> /* status_t */

#include <boost/io/ios_state.hpp>

#include <sys/uio.h> /* iovec */

#include <algorithm> /* opy */
#include <chrono> /* seconds */
#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <iomanip> /* hex */
#include <memory> /* make_shared, shared_ptr */
#include <string>
#include <system_error>
#include <thread> /* sleep_for */
#include <tuple>

component::IFabric_client * open_connection_patiently(component::IFabric_endpoint_unconnected_client *aep_)
{
	component::IFabric_client *cnxn = nullptr;
	int try_count = 0;
	while ( ! cnxn )
	{
		try
		{
			cnxn = aep_->make_open_client();
		}
		catch ( std::system_error &e )
		{
			if ( e.code().value() != ECONNREFUSED )
			{
				throw;
			}
		}
		++try_count;
	}
	assert(0U < cnxn->max_message_size());
	return cnxn;
}

namespace cw
{
	struct remote_memory_client
		: fi_context2
	{
	private:
		std::unique_ptr<component::IFabric_endpoint_unconnected_client> _ep;
		std::shared_ptr<cw::registered_memory> _rm_out;
		std::shared_ptr<cw::registered_memory> _rm_in;
		::iovec _v[1];
		cw::registered_memory &rm_in() const { return *_rm_in; }
	public:
		remote_memory_client(
			component::IFabric &fabric
			, const std::string &fabric_spec
			, const std::string ip_address
			, std::uint16_t port
			, std::size_t memory_size
			, std::uint64_t remote_key_base
		);
		remote_memory_client(remote_memory_client &&) noexcept = default;
		remote_memory_client &operator=(remote_memory_client &&) noexcept = default;

		std::shared_ptr<cw::registered_memory> rm_in_raw() const { return _rm_in; }
		std::shared_ptr<cw::registered_memory> rm_out_raw() const { return _rm_out; }
		component::IFabric_client * open_connection() { return open_connection_patiently(_ep.get()); }
		cw::registered_memory &rm_out() const { return *_rm_out; }
	};
}

cw::remote_memory_client::remote_memory_client(
	component::IFabric &fabric_
	, const std::string &fabric_spec_
	, const std::string ip_address_
	, std::uint16_t port_
	, std::size_t memory_size_
	, std::uint64_t remote_key_base_
)
try
	: _ep(fabric_.make_endpoint(fabric_spec_, ip_address_, port_))
	, _rm_out{
		std::make_shared<cw::registered_memory>(
			_ep.get()
#if 0
			, memory_size_
#else
			, std::make_unique<cw::dram_memory>(memory_size_)
#endif
			, remote_key_base_ * 2U
		)
	}
	, _rm_in{
		std::make_shared<cw::registered_memory>(
			_ep.get()
#if 0
			, memory_size_
#else
			, std::make_unique<cw::dram_memory>(memory_size_)
#endif
			, remote_key_base_ * 2U + 1
		)
	}
	, _v{::iovec{&rm_out()[0], sizeof(uint64_t) + sizeof (uint64_t)}}
{
	_ep->post_recv(_v, this);
}
catch ( std::exception &e )
{
	FLOGM("{}", e.what());
	throw;
}

namespace
{
	template <typename D>
		double double_seconds(D d_)
		{
			return std::chrono::duration<double>(d_).count();
		}

	template <typename C = std::chrono::steady_clock>
		struct timer
		{
			using clock_type = C;
		private:
			typename clock_type::time_point _t;
		public:
			timer()
				: _t(clock_type::now())
			{}
			typename clock_type::duration elapsed() const { return clock_type::now() - _t; }
		};

	struct ct
		: public common::hist_log2
	{
	private:
		std::string _tag;

	public:
		ct(common::string_view tag_)
			: common::hist_log2()
			, _tag(tag_)
		{}

		void record(double t)
		{
			common::hist_log2::record(t*1e6);
		}
	};
} // namespace

int main(int, char **argv)
{
	/* server invocation:
	 *   fabric-test1
	 * client in invocation:
	 *   fabric-test1 10.0.0.91
	 */

	assert( argv[1] );

#if CW_TEST
	const char *remote_host = argv[1];
	
	fabric_component fc;
	fabric_factory fa(fc);
	fabric_fabric ff(fa);

	/* Cannot write the whole memory because
	 *  (1) remote memory also have a small offset for an eyecatcher.
	 */
	std::string msg(cw::test_data::size(), 'H');
	/* allow time for the server to listen before the client restarts.
	 * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
	 * state, causing server "bind" to fail with EADDRINUSE.
	 */
	std::this_thread::sleep_for(std::chrono::milliseconds(2500));
	FLOG("CLIENT begin port {}", fabric_fabric::control_port);

	auto remote_key_index = 0;
	/* An ordinary client which tests RDMA to server memory */
	cw::remote_memory_client
		uc(
			*ff.fabric()
			, empty_object_json.str()
			, remote_host
			, fabric_fabric::control_port
			, cw::test_data::memory_size()
			, remote_key_index
		);

	cw::remote_memory_client_connected
		client(
			uc.rm_out_raw()
			, uc.rm_in_raw()
			, cw::get_rm(uc.open_connection(), &uc, &uc.rm_out()[0])
			, cw::quit_option::do_quit
		);

	auto t_start = std::chrono::steady_clock::now();
	/* expect that all max messages are greater than 0, and the same */
	assert(0U < client.max_message_size());

	ct ct_w("wr");
	ct ct_r("rd");

	unsigned longish_write = 0;
	unsigned long_write = 0;
	unsigned longish_read = 0;
	unsigned long_read = 0;

	const std::size_t count = common::env_value<std::size_t>("COUNT", 10000);
	FLOG("size {} count {}", cw::test_data::size(), count);

	for ( auto iter1 = 0U; iter1 != count; ++iter1 )
	{
		{
			timer t_write;
#if 1
			auto st = client.read(cw::test_data::size());
#elif 0
			auto st = client.write(msg);
#else
			auto st = client.write_uninitialized(cw::test_data::size());
#endif
			(void) st;
			assert(st == ::S_OK);
			auto tm = double_seconds(t_write.elapsed());
			ct_w.record(tm);
			longish_write += ( 1.0 <= tm );
			long_write += ( 2.0 <= tm );
			if ( 1.0 <= tm ) FLOG("longish write ({}) {}s", iter1, tm);
		}
		{
#if 0
			timer t_read;
			client.read_verify(msg);
			auto t = double_seconds(t_read.elapsed());
			++ct_r.at(unsigned(std::log2(t*1e6))); /* element 0 is [1..2) microsecond */
			longish_read += ( 1.0 <= t );
			long_read += ( 2.0 <= t );
			if ( 1.0 <= t ) FLOG("longish read ({}) {}s", iter1, t);
#endif
		}
		/* client destructor sends FI_SHUTDOWN to server */
	}
	auto t_duration = std::chrono::steady_clock::now() - t_start;

	auto data_size_total = uint64_t(count) * msg.size();
	FLOG("Data rate {} bytes in {} seconds {} GB/sec lw {}/{}/{}/{}", data_size_total, double_seconds(t_duration), double(data_size_total) / 1e9 / double_seconds(t_duration), longish_write, long_write, longish_read, long_read);
	FLOG("wr {}", ct_w.out("usec"));

#else
	(void)argv;
#endif
	return 0;
}
