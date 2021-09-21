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

#include "eyecatcher.h"
#include "registered_memory.h"
#include "quit_option.h"
#include "server_connection.h"

#include <api/components.h>
#include <api/fabric_itf.h> /* IFabric, IFabric_client, IFabric_server_factory, IFabric_endpoint_connected, fi_context2 */

#include <common/env.h>
#include <common/errors.h> /* S_OK */
#include <common/histogram.h>
#include <common/json.h>
#include <common/logging.h>
#include <common/string_view.h>
#include <common/types.h> /* status_t */

#include <boost/core/noncopyable.hpp>

#include <sys/uio.h> /* iovec */

#include <algorithm> /* max, copy */
#include <chrono> /* seconds */
#include <cmath> /* log2 */
#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t, uint64_t */
#include <cstdlib> /* getenv */
#include <cstring> /* memcpy */
#include <exception>
#include <functional> /* function, ref */
#include <future>
#include <iomanip> /* hex */
#include <memory> /* make_shared, shared_ptr */
#include <numeric> /* accumulate */
#include <string>
#include <system_error>
#include <thread> /* sleep_for */
#include <vector>

namespace component
{
	class IFabric_endpoint_unconnected_client;
	class IFabric_client;
}

component::IFabric_client * open_connection_patiently(component::IFabric_endpoint_unconnected_client *aep);

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

namespace component
{
	class IFabric;
	class IFabric_endpoint_unconnected_client;
	class IFabric_client;
	class IFabric_endpoint_comm;
}

struct registered_memory;

/*
 * returns: number of polls (including the successful poll)
 */
unsigned wait_poll(
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


namespace component
{
	class IFabric_endpoint_connected;
}

struct registered_memory;

struct remote_memory_accessor
	: fi_context2
{
protected:
	remote_memory_accessor()
	{}
	void send_memory_info(component::IFabric_endpoint_connected &cnxn, registered_memory &rm);
public:
	/* using rm as a buffer, send message */
	void send_msg(component::IFabric_endpoint_connected &cnxn, registered_memory &rm, const void *msg, std::size_t len);
};

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
		::wait_poll(
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
		FLOGM("exception {} {}", e.what(), eyecatcher);
	}
}
struct cw_remote_memory_client
	: public remote_memory_accessor
{
private:
	static void check_complete_static(void *t, void *ctxt, ::status_t stat, std::size_t len);
	void check_complete(::status_t stat, std::size_t len);

	std::unique_ptr<component::IFabric_endpoint_unconnected_client> _ep;
	std::shared_ptr<registered_memory> _rm_out;
	std::shared_ptr<registered_memory> _rm_in;
	::iovec _v[1];
	std::shared_ptr<component::IFabric_client> _cnxn;
	std::uint64_t _vaddr;
	std::uint64_t _key;
	quit_option _quit_flag;
	::status_t _last_stat;

	registered_memory &rm_in() const { return *_rm_in; }
	registered_memory &rm_out() const { return *_rm_out; }
	status_t wait_complete();
public:
	cw_remote_memory_client(
		component::IFabric &fabric
		, const std::string &fabric_spec
		, const std::string ip_address
		, std::uint16_t port
		, std::size_t memory_size
		, std::uint64_t remote_key_base
	, quit_option quit_flag = quit_option::do_not_quit
	);
	cw_remote_memory_client(cw_remote_memory_client &&) noexcept = default;
	cw_remote_memory_client &operator=(cw_remote_memory_client &&) noexcept = default;

	~cw_remote_memory_client();

	void send_disconnect(component::IFabric_endpoint_comm &cnxn_, registered_memory &rm_, quit_option quit_flag_);

	std::uint64_t vaddr() const { return _vaddr; }
	std::uint64_t key() const { return _key; }
	component::IFabric_client &cnxn() { return *_cnxn; }

	status_t write(common::string_view msg);
	status_t write_uninitialized(std::size_t s);

	status_t read(std::size_t s);
	status_t read_verify(common::string_view msg_);
	std::size_t max_message_size() const
	{
		return _cnxn->max_message_size();
	}
};

void cw_remote_memory_client::check_complete_static(void *t_, void *ctxt_, ::status_t stat_, std::size_t len_)
try
{
	/* The callback context must be the object which was polling. */
	(void)t_;
	assert(t_ == ctxt_);
	auto rmc = static_cast<cw_remote_memory_client *>(ctxt_);
	assert(rmc);
	rmc->check_complete(stat_, len_);
}
catch ( std::exception &e )
{
	FLOGF("exception {}", e.what());
}

void cw_remote_memory_client::check_complete(::status_t stat_, std::size_t)
{
	_last_stat = stat_;
}

cw_remote_memory_client::cw_remote_memory_client(
	component::IFabric &fabric_
	, const std::string &fabric_spec_
	, const std::string ip_address_
	, std::uint16_t port_
	, std::size_t memory_size_
	, std::uint64_t remote_key_base_
	, quit_option quit_flag_
)
try
	: remote_memory_accessor()
	, _ep(fabric_.make_endpoint(fabric_spec_, ip_address_, port_))
	, _rm_out{std::make_shared<registered_memory>(*_ep, memory_size_, remote_key_base_ * 2U)}
	, _rm_in{std::make_shared<registered_memory>(*_ep, memory_size_, remote_key_base_ * 2U + 1)}
	, _v{::iovec{&rm_out()[0], (sizeof _vaddr) + (sizeof _key)}}
	, _cnxn((_ep->post_recv(_v, this), open_connection_patiently(_ep.get())))
	, _vaddr{}
	, _key{}
	, _quit_flag(quit_flag_)
	, _last_stat(::E_FAIL)
{
	/* The server's first action should be to send us the remote memory address and key */
	::wait_poll(
		*_cnxn
		, [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
			{
				(void)ctxt_;
				(void)stat_;
				(void)len_;
				assert(ctxt_ == this);
				assert(stat_ == ::S_OK);
				assert(len_ == (sizeof _vaddr) + sizeof( _key));
				std::memcpy(&_vaddr, &rm_out()[0], sizeof _vaddr);
				std::memcpy(&_key, &rm_out()[sizeof _vaddr], sizeof _key);
			}
	);
	FLOG("Client: remote memory addr {} key {:x}", reinterpret_cast<void*>(_vaddr), std::hex);
}
catch ( std::exception &e )
{
	FLOGM("{}", e.what());
	throw;
}

void cw_remote_memory_client::send_disconnect(component::IFabric_endpoint_comm &cnxn_, registered_memory &rm_, quit_option quit_flag_)
{
	send_msg(cnxn_, rm_, &quit_flag_, sizeof quit_flag_);
}

cw_remote_memory_client::~cw_remote_memory_client()
{
	if ( _cnxn )
	{
		try
		{
			send_disconnect(cnxn(), rm_out(), _quit_flag);
		}
		catch ( std::exception &e )
		{
			FLOGM("exception {} {}", e.what(), eyecatcher);
		}
	}
}

status_t cw_remote_memory_client::wait_complete()
{
	::wait_poll(
		*_cnxn
		, [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *)
			{
				check_complete_static(this, ctxt_, stat_, len_);
			}
	);
	if ( _last_stat != ::S_OK )
	{
		FLOGM(": {}", _last_stat);
	}
	return _last_stat;
}

status_t cw_remote_memory_client::write(const common::string_view msg_)
{
	std::copy(msg_.begin(), msg_.end(), &rm_out()[0]);
	return write_uninitialized(msg_.size());
}

status_t cw_remote_memory_client::write_uninitialized(std::size_t sz_)
{
	::iovec buffers[1] = { ::iovec{ &rm_out()[0], sz_ } };
	_cnxn->post_write(buffers, _vaddr + remote_memory_offset, _key, this);
	return wait_complete();
}

status_t cw_remote_memory_client::read(std::size_t sz_)
{
	::iovec buffers[1] = { ::iovec{ &rm_in()[0], sz_ } };
	_cnxn->post_read(buffers, _vaddr + remote_memory_offset, _key, this);
	return wait_complete();
}

status_t cw_remote_memory_client::read_verify(const common::string_view msg_)
{
	auto st =  read(msg_.size());
	std::string remote_msg(&rm_in()[0], &rm_in()[0] + msg_.size());
	assert(msg_ == remote_msg);
	return st;
}

namespace component
{
	class IFabric;
	class IFabric_server_factory;
}

/*
 * A component::IFabric_server_factory, which will support clients until one
 * of them closes with the "quit" flag set.
 */
struct cw_remote_memory_server
	: public remote_memory_accessor
	, private boost::noncopyable
{
private:
	std::shared_ptr<component::IFabric_server_factory> _ep;
	std::future<void> _th;

	void listener(
		component::IFabric_server_factory &ep
		, std::size_t memory_size
		, std::uint64_t remote_key_index
	);

	void listener_counted(
		component::IFabric_server_factory &ep
		, std::uint64_t remote_key_index
		, std::size_t memory_size
		, unsigned cnxn_count
	);
public:
	cw_remote_memory_server(
		component::IFabric &fabric
		, const std::string &fabric_spec
		, std::uint16_t control_port
		, const char *
		, std::size_t memory_size
		, std::uint64_t remote_key_base
	);
	cw_remote_memory_server(
		component::IFabric &fabric
		, const std::string &fabric_spec
		, std::uint16_t control_port
		, const char *
		, std::size_t memory_size
		, std::uint64_t remote_key_base
		, unsigned cnxn_limit
	);
	~cw_remote_memory_server();
	std::size_t max_message_size() const;
};

namespace component
{
	class IFabric_server_factory;
}

struct server_connection_and_memory
	: public server_connection
	, public registered_memory
	, public remote_memory_accessor
	, private boost::noncopyable
{
	server_connection_and_memory(
		component::IFabric_server_factory &ep
		, std::size_t memory_size
		, std::uint64_t remote_key
	);
	~server_connection_and_memory();
};

server_connection_and_memory::server_connection_and_memory(
	component::IFabric_server_factory &ep_
	, std::size_t memory_size_
	, std::uint64_t remote_key_
)
	: server_connection(ep_)
	, registered_memory(cnxn(), memory_size_, remote_key_)
	, remote_memory_accessor()
{
	/* send the address, and the key to memory */
	send_memory_info(cnxn(), *this);
}

server_connection_and_memory::~server_connection_and_memory()
{
	std::vector<::iovec> v;
	::iovec iv;
	iv.iov_base = &((*this)[0]);
	iv.iov_len = 1;
	try
	{
		v.emplace_back(iv);
		cnxn().post_recv(v, this);
		wait_poll(
			cnxn()
			, [this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
				{
					(void)ctxt_;
					(void)stat_;
					(void)len_;
					assert(ctxt_ == this);
					assert(stat_ == S_OK);
					assert(len_ == 1);
				}
		);
	}
	catch ( std::exception &e )
	{
		FLOGM("{}", e.what());
	}
}

void cw_remote_memory_server::listener(
	component::IFabric_server_factory &ep_
	, std::size_t memory_size_
	, std::uint64_t remote_key_index_
)
{
	auto quit = false;
	for ( ; ! quit; ++remote_key_index_ )
	{
		server_connection sc(ep_);
		assert(sc.cnxn().max_message_size() == this->max_message_size());
		/* register an RDMA memory region */
		registered_memory rm{sc.cnxn(), memory_size_, remote_key_index_};
		try
		{
			/* set a receive buffer. We will get one message, when the client is done with rm */
			::iovec v[1] = { ::iovec{&rm[0], 1} };
			sc.cnxn().post_recv(v, this);
			/* send the client address and key to memory */
			send_memory_info(sc.cnxn(), rm);
			/* wait for client indicate exit (by sending one byte to us) */
			::wait_poll(
				sc.cnxn()
				, [&quit, &rm, this] (void *ctxt_, ::status_t stat_, std::uint64_t, std::size_t len_, void *) -> void
					{
						(void)ctxt_;
						(void)stat_;
						(void)len_;
						assert(ctxt_ == this);
						assert(stat_ == S_OK);
						assert(len_ == 1);
						/* did client leave with the "quit byte" set to 'q'? */
						quit |= rm[0] == char(quit_option::do_quit);
					}
			);
		}
		catch ( std::exception &e )
		{
			FLOGM("{}",  e.what());
			throw;
		}
	}
}

void cw_remote_memory_server::listener_counted(
	component::IFabric_server_factory &ep_
	, std::size_t memory_size_
	, std::uint64_t remote_key_index_
	, unsigned cnxn_count_
)
{
	std::vector<std::shared_ptr<server_connection_and_memory>> scrm;
	for ( auto i = 0U; i != cnxn_count_; ++i )
	{
		scrm.emplace_back(std::make_shared<server_connection_and_memory>(ep_, memory_size_, remote_key_index_ + i));
	}
}

cw_remote_memory_server::cw_remote_memory_server(
	component::IFabric &fabric_
	, const std::string &fabric_spec_
	, std::uint16_t control_port_
	, const char *
	, std::size_t memory_size_
	, std::uint64_t remote_key_base_
)
	: remote_memory_accessor()
	, _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
	, _th(
			std::async(
				std::launch::async
				, &cw_remote_memory_server::listener
				, this
				, std::ref(*_ep)
				, memory_size_
				, remote_key_base_
			)
		)
{
}

cw_remote_memory_server::cw_remote_memory_server(
	component::IFabric &fabric_
	, const std::string &fabric_spec_
	, std::uint16_t control_port_
	, const char *
	, std::size_t memory_size_
	, std::uint64_t remote_key_base_
	, unsigned cnxn_limit_
)
	: _ep(fabric_.open_server_factory(fabric_spec_, control_port_))
	, _th(
			std::async(
				std::launch::async
				, &cw_remote_memory_server::listener_counted
				, this
				, std::ref(*_ep)
				, memory_size_
				, remote_key_base_
				, cnxn_limit_
			)
		)
{
}

cw_remote_memory_server::~cw_remote_memory_server()
{
	try
	{
		_th.get();
	}
	catch ( std::exception &e )
	{
		FLOGM("exception {} {}", e.what(), eyecatcher);
	}
}

std::size_t cw_remote_memory_server::max_message_size() const
{
	return _ep->max_message_size();
}

// The fixture for testing class Foo.
class Fabric_test
{
	static const char *control_port_spec;
protected:
	// Objects declared here can be used by all tests in the test case

	static const std::uint16_t control_port;
	static const std::size_t data_size;
	static const std::size_t count;
	static const std::size_t memory_size;

	static std::string fabric_spec() {
		namespace c_json = common::json;
		using json = c_json::serializer<c_json::dummy_writer>;

		auto mr_mode =
			json::array(
				"FI_MR_LOCAL"
				, "FI_MR_VIRT_ADDR"
				, "FI_MR_ALLOCATED"
				, "FI_MR_PROV_KEY"
			);

		auto domain_name_spec = json::object();

		auto fabric_attr =
			json::member
			( "fabric_attr"
				, json::object(json::member("prov_name", json::string("verbs")))
			);
		auto domain_attr =
			json::member(
				"domain_attr"
				, std::move(
						json::object(
							json::member("mr_mode", std::move(mr_mode))
							, json::member("threading", "FI_THREAD_SAFE")
						).append(std::move(domain_name_spec))
					)
			);
		auto ep_attr =
			json::member(
				"ep_attr"
				, json::object(json::member("type", "FI_EP_MSG"))
			);
		return json::object(std::move(fabric_attr), std::move(domain_attr), std::move(ep_attr)).str();
	}

	/* create object instance through factory */
	gsl::not_null<component::IBase *> _comp = component::load_component(
		"libcomponent-fabric.so"
		, component::net_fabric_factory
	);

	component::Itf_ref<component::IFabric_factory> _factory;
	std::unique_ptr<component::IFabric> _fabric;
public:
	void WriteReadSequential();
	Fabric_test()
		: _comp(
			component::load_component(
				"libcomponent-fabric.so"
				, component::net_fabric_factory
			)
		)
		, _factory(component::make_itf_ref(static_cast<component::IFabric_factory *>(_comp->query_interface(component::IFabric_factory::iid()))))
		, _fabric(_factory->make_fabric(fabric_spec()))
	{}
};

const std::uint16_t Fabric_test::control_port = common::env_value<std::uint16_t>("FABRIC_TEST_CONTROL_PORT", 47591);
const std::size_t Fabric_test::data_size = common::env_value<std::size_t>("SIZE", 1U<<23);
const std::size_t Fabric_test::count = common::env_value<std::size_t>("COUNT", 10000);
const std::size_t Fabric_test::memory_size = Fabric_test::data_size + std::max(remote_memory_offset, std::size_t(100));

std::ostream &describe_ep(std::ostream &o_, const component::IFabric_server_factory &e_)
{
	return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

namespace
{
	auto empty_object_json = common::json::serializer<common::json::dummy_writer>::object{};

static constexpr auto count_outer = 1U;

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

void write_read_sequential_client(
	component::IFabric & fabric_
	, const uint16_t control_port_
	, const char *const remote_host_
	, std::size_t memory_size
	, std::size_t data_size
	, std::size_t count
)
{
	/* Cannot write the whole memory because
	 *  (1) registered_memory writes to a small offset to force bad alignment in local memory (for testing) and
	 *  (2) remote memory also have a small offset for an eyecatcher.
	 * Neither of these are essential. If we were simply writing a performance test, wuold get rid of both.
	 */
	std::string msg(data_size, 'H');
	/* allow time for the server to listen before the client restarts.
	 * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
	 * state, causing server "bind" to fail with EADDRINUSE.
	 */
	std::this_thread::sleep_for(std::chrono::milliseconds(2500));
	FLOG("CLIENT begin port {}", control_port_);

	unsigned longish_write = 0;
	unsigned long_write = 0;
	unsigned longish_read = 0;
	unsigned long_read = 0;


	auto remote_key_index = 0;
	/* An ordinary client which tests RDMA to server memory */
	cw_remote_memory_client client(
			fabric_
			, empty_object_json.str()
			, remote_host_ , control_port_
			, memory_size
			, remote_key_index
			, quit_option::do_quit
		);

	auto t_start = std::chrono::steady_clock::now();
	/* expect that all max messages are greater than 0, and the same */
	assert(0U < client.max_message_size());

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

	ct ct_w("wr");
	ct ct_r("rd");

	for ( auto iter1 = 0U; iter1 != count; ++iter1 )
	{
		{
			timer t_write;
#if 1
			auto st = client.read(data_size);
#elif 0
			auto st = client.write(msg);
#else
			auto st = client.write_uninitialized(data_size);
#endif
			(void) st;
			assert(st == ::S_OK);
			auto t = double_seconds(t_write.elapsed());
			ct_w.record(t);
			longish_write += ( 1.0 <= t );
			long_write += ( 2.0 <= t );
			if ( 1.0 <= t ) FLOG("longish write ({}) {}s", iter1, t);
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
	FLOG("Data rate {} bytes in {} seconds {} GB/sec lw {}/{} lr {}/{}, histogram base 1 usec", data_size_total, double_seconds(t_duration), double(data_size_total) / 1e9 / double_seconds(t_duration), longish_write, long_write, longish_read, long_read);
	FLOG("wr {}", ct_w.out("usec"));

	/* In case the provider actually uses the remote keys which we provide, make them unique.
	 * (At server shutdown time there are no other clients, so the shutdown client may use any value.)
	 */
	remote_key_index = 0U;
	/* A special client to tell the server factory to shut down. Placed after other clients because the server apparently cannot abide concurrent clients. */
}

void write_read_sequential_server(component::IFabric & fabric_, const uint16_t control_port_, const std::size_t memory_size_)
{
	FLOG("SERVER begin port {}", control_port_);
	{
		auto remote_key_base = 0U;
		cw_remote_memory_server server(fabric_, empty_object_json.str(), control_port_, "", memory_size_, remote_key_base);
		assert(0U < server.max_message_size());
	}
	FLOG("SERVER end");
}

} // namespace

void Fabric_test::WriteReadSequential()
{
	FLOG("size {} count {}", data_size, count);
	const char *remote_host = ::getenv("SERVER");
	if ( remote_host )
	{
		write_read_sequential_client(*_fabric, control_port, remote_host, memory_size, data_size, count);
	}
	else
	{
		write_read_sequential_server(*_fabric, control_port, memory_size);
	}
}

int main()
{
	/* server invocation:
	 *   fabric-test1
	 * client in invocation:
	 *   SERVER=10.0.0.91 fabric-test1
	 */

	Fabric_test t;

	t.WriteReadSequential();

	return 0;
}
