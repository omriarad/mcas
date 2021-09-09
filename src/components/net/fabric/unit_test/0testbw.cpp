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

#include <api/components.h>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <common/json.h>
#include <common/logging.h>

#include <api/fabric_itf.h>

#include "eyecatcher.h"
#include "patience.h" /* open_connection_patiently */
#include "registration.h"
#include "remote_memory_server.h"
#include "remote_memory_subserver.h"
#include "remote_memory_subclient.h"
#include "remote_memory_client.h"

#include <sys/time.h>
#include <sys/resource.h>

#include <algorithm> /* max, min */
#include <chrono> /* seconds */
#include <cinttypes> /* PRIu64 */
#include <cstdlib> /* getenv */
#include <cstring> /* strpbrk */
#include <exception>
#include <stdexcept> /* domain_error */
#include <memory> /* shared_ptr */
#include <iostream> /* cerr */
#include <future> /* async, future */
#include <thread> /* sleep_for */

// The fixture for testing class Foo.
class Fabric_test : public ::testing::Test
{
  static const char *control_port_spec;
protected:
  // Objects declared here can be used by all tests in the test case

  static const std::uint16_t control_port_0;
  static const std::uint16_t control_port_1;
  static const std::uint16_t control_port_2;

  static std::string fabric_spec(const std::string &prov_name_) {
    namespace c_json = common::json;
    using json = c_json::serializer<c_json::dummy_writer>;

    auto verbs_mr_mode =
      json::array(
        "FI_MR_LOCAL"
        , "FI_MR_VIRT_ADDR"
        , "FI_MR_ALLOCATED"
        , "FI_MR_PROV_KEY"
      );

    /* Although man fi_mr says "FI_MR_BASIC is maintained for backwards
     * compatibility (libfabric version 1.4 or earlier)", sockets as of 1.6
     * will not accept the newer, explicit list.
     *
     * Although man fi_mr says "providers that support basic registration
     * usually required FI_MR_LOCAL", the socket provider will not accept
     * FI_MR_LOCAL.
     */
    auto sockets_mr_mode =
      json::array("FI_MR_BASIC")
      ;

	auto domain_name_verbs_spec =
      domain_name_verbs
      ? json::object(json::member("name", domain_name_verbs))
      : json::object()
      ;

	auto domain_name_sockets_spec =
      domain_name_sockets
      ? json::object(json::member("name", domain_name_sockets))
      : json::object()
      ;

	auto domain_name_spec =
      std::move(prov_name_ == std::string("sockets") ? domain_name_sockets_spec : domain_name_verbs_spec);

    auto mr_mode = std::move(prov_name_ == std::string("sockets") ? sockets_mr_mode : verbs_mr_mode);

    auto fabric_attr =
      json::member
      ( "fabric_attr"
        , json::object(json::member("prov_name", json::string(prov_name_)))
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
  static const char *remote_host;

  static bool is_client() { return bool(remote_host); }
  static bool is_server() { return ! is_client(); }
public:
  static const char *domain_name_verbs;
  static const char *domain_name_sockets;

};

const char *Fabric_test::control_port_spec = std::getenv("FABRIC_TEST_CONTROL_PORT");
const std::uint16_t Fabric_test::control_port_0 = uint16_t(control_port_spec ? std::strtoul(control_port_spec, 0, 0) : 47591);
const std::uint16_t Fabric_test::control_port_1 = uint16_t(control_port_0 + 1);
const std::uint16_t Fabric_test::control_port_2 = uint16_t(control_port_0 + 2);


const char *Fabric_test::domain_name_verbs = ::getenv("DOMAIN_VERBS");
const char *Fabric_test::domain_name_sockets = ::getenv("DOMAIN_SOCKETS");
const char *Fabric_test::remote_host = ::getenv("SERVER");

std::ostream &describe_ep(std::ostream &o_, const component::IFabric_server_factory &e_)
{
  return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

namespace
{
  auto empty_object_json = common::json::serializer<common::json::dummy_writer>::object{};

static constexpr auto count_outer = 3U;
static constexpr auto count_inner = 3U;
static constexpr std::size_t memory_size = 409600000;
static constexpr unsigned long iterations = 1000;

template <typename D>
  double double_seconds(D d_)
  {
    return std::chrono::duration<double>(d_).count();
  }

void write_read_sequential_client(component::IFabric & fabric_, const unsigned iter0_, const uint16_t control_port_, const char *const remote_host_, bool force_error_) // const std::string &fabric_spec_, const char *const remote_host, uint16_t control_port_, bool force_error_
{
	/* Cannot write the whole memory because 
	 *  (1) registered_memory writes to a small offset to force bad alignment in local memory (for testing) and
	 *  (2) remote memory also have a small offset for an eyecatcher.
	 * Neither of these are essential. If we were simple wrinting a performance test, wuold get rid of both.
	 */
	std::string msg(memory_size-std::max(remote_memory_offset, std::size_t(100)), 'H');
	/* allow time for the server to listen before the client restarts.
	 * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
	 * state, causing server "bind" to fail with EADDRINUSE.
	 */
	std::this_thread::sleep_for(std::chrono::milliseconds(2500));
	auto t_start = std::chrono::steady_clock::now();
	std::cerr << "CLIENT begin " << iter0_ << " port " << control_port_ << std::endl;

	/* In case the provider actually uses the remote keys which we provide, make them unique. */
	auto remote_key_index = iter0_;
	/* An ordinary client which tests RDMA to server memory */
	remote_memory_client client(
		test_type::performance
		, fabric_
		, empty_object_json.str()
		, remote_host_ , control_port_
		, memory_size , remote_key_index
		, quit_option::do_quit
	);

	/* expect that all max messages are greater than 0, and the same */
	EXPECT_LT(0U, client.max_message_size());
	for ( auto iter1 = 0U; iter1 != count_inner; ++iter1 )
	{
#if 0
		/* Feed the server_factory some terrible text */
		std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";
#endif
		client.write(msg, force_error_);
		if ( ! force_error_ )
		{
			client.read_verify(msg);
		}
		/* client destructor sends FI_SHUTDOWN to server */
		std::cerr << "CLIENT end " << iter0_ << std::endl;
	}
	auto t_duration = std::chrono::steady_clock::now() - t_start;

	auto data_size = uint64_t(count_inner) * msg.size();
	std::cerr << "Data rate " << std::dec << data_size << " bytes in " << double_seconds(t_duration) << " seconds " << double(data_size) / 1e9 / double_seconds(t_duration) << " GB/sec" << "\n";

	/* In case the provider actually uses the remote keys which we provide, make them unique.
	 * (At server shutdown time there are no other clients, so the shutdown client may use any value.)
	 */
	 remote_key_index = 0U;
}

void write_read_sequential_server(component::IFabric & fabric_, const unsigned iter0_, const uint16_t control_port_) // (const std::string &fabric_spec_, const char *const remote_host, uint16_t control_port_, bool force_error_)
{
	std::cerr << "SERVER begin " << iter0_ << " port " << control_port_ << std::endl;
	{
		auto remote_key_base = 0U;
		remote_memory_server server(test_type::performance, fabric_, empty_object_json.str(), control_port_, "", memory_size, remote_key_base);
		EXPECT_LT(0U, server.max_message_size());
	}
	std::cerr << "SERVER end " << iter0_ << std::endl;
}

void write_read_sequential(const std::string &fabric_spec_, const char *const remote_host, uint16_t control_port_, bool force_error_)
{
	for ( auto iter0 = 0U; iter0 != count_outer; ++iter0 )
	{
		/* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
		 * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
		 * Shouldn't the client see some sort of error?
		 */
		auto control_port = std::uint16_t(control_port_ + iter0);
		/* create object instance through factory */
		component::IBase * comp = component::load_component("libcomponent-fabric.so",
                                                        component::net_fabric_factory);
		ASSERT_TRUE(comp);

		auto factory = make_itf_ref(static_cast<component::IFabric_factory *>(comp->query_interface(component::IFabric_factory::iid())));
		auto fabric = std::shared_ptr<component::IFabric>(factory->make_fabric(fabric_spec_));
		if ( remote_host )
		{
			write_read_sequential_client(*fabric, iter0, control_port, remote_host, force_error_);
		}
		else
		{
			write_read_sequential_server(*fabric, iter0, control_port);
		}
	}
}

TEST_F(Fabric_test, WriteReadSequential)
{
  write_read_sequential(fabric_spec("verbs"), remote_host, control_port_2, false);
}

#if 0
TEST_F(Fabric_test, WriteReadParallel)
{
  for ( auto iter0 = 0U; iter0 != count_outer; ++iter0 )
  {
    /* To avoid conflicts with server which are slow to shut down, use a different control port on every pass
     * But, what happens if a server is shut down (fi_shutdown) while a client is expecting to receive data?
     * Shouldn't the client see some sort of error?
     */
    auto control_port = std::uint16_t(control_port_2 + iter0);
    /* create object instance through factory */
    component::IBase * comp = component::load_component("libcomponent-fabric.so",
                                                        component::net_fabric_factory);
    ASSERT_TRUE(comp);

    auto factory = component::make_itf_ref(static_cast<component::IFabric_factory *>(comp->query_interface(component::IFabric_factory::iid())));

    auto fabric = std::shared_ptr<component::IFabric>(factory->make_fabric(fabric_spec("verbs")));
    if ( is_server() )
    {
      std::cerr << "SERVER begin " << iter0 << " port " << control_port << std::endl;
      {
        auto expected_client_count = count_inner;
        auto remote_key_base = 0U;
        remote_memory_server server(*fabric, empty_object_json.str(), control_port, "", memory_size, remote_key_base, expected_client_count);
        EXPECT_LT(0U, server.max_message_size());
      }
      std::cerr << "SERVER end " << iter0 << std::endl;
    }
    else
    {
      /* allow time for the server to listen before the client restarts.
       * For an unknown reason, client "connect" puts the port in "ESTABSLISHED"
       * state, causing server "bind" to fail with EADDRINUSE.
       */
      std::this_thread::sleep_for(std::chrono::milliseconds(2500));
      {
        std::vector<remote_memory_client> vv;
        for ( auto iter1 = 0U; iter1 != count_inner; ++iter1 )
        {
          /* In case the provider actually uses the remote keys which we provide, make them unique. */
          auto remote_key_index = iter1;

          /* Ordinary clients which test RDMA to server memory.
           * Should be able to control them via pointers or (using move semantics) in a vector of objects.
           */
          vv.emplace_back(*fabric, empty_object_json.str(), remote_host, control_port, memory_size, remote_key_index);
          /* expect that all max messages are greater than 0, and the same */
          EXPECT_LT(0U, vv.back().max_message_size());
          EXPECT_EQ(vv.front().max_message_size(), vv.back().max_message_size());
        }

        /* Feed the server_factory some terrible text */
        std::string msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.";

        for ( auto &client : vv )
        {
          client.write(msg);
        }

        for ( auto &client : vv )
        {
          client.read_verify(msg);
        }
        /* client destructors send FI_SHUTDOWNs to server */
      }
      /* The remote_memory servr will shut down after it has seen a specified number of clients. */
    }
  }
}
#endif
} // namespace

int main(int argc, char **argv)
{
  /* server invocation:
   *   fabric-test1
   * client in invocation:
   *   SERVER=10.0.0.91 fabric-test1
   */

  std::cerr << "Domain/verbs is " << (Fabric_test::domain_name_verbs ? Fabric_test::domain_name_verbs : "unspecified") << "\n";
  std::cerr << "Domain/sockets is " << (Fabric_test::domain_name_sockets ? Fabric_test::domain_name_sockets : "unspecified") << "\n";

  ::testing::InitGoogleTest(&argc, argv);
  auto r = RUN_ALL_TESTS();

  return r;
}
