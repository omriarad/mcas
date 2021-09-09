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
#include "pingpong_client.h"
#include "pingpong_server.h"
#include "pingpong_server_n.h"
#include "pingpong_stat.h"
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

	auto domain_name_verbs_spec =
		domain_name_verbs
		? json::object(json::member("name", domain_name_verbs))
		: json::object()
		;

	auto domain_name_spec = std::move(domain_name_verbs_spec);

	auto mr_mode = std::move(verbs_mr_mode);

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

};

const char *Fabric_test::control_port_spec = std::getenv("FABRIC_TEST_CONTROL_PORT");
const std::uint16_t Fabric_test::control_port_0 = uint16_t(control_port_spec ? std::strtoul(control_port_spec, 0, 0) : 47591);
const std::uint16_t Fabric_test::control_port_1 = uint16_t(control_port_0 + 1);
const std::uint16_t Fabric_test::control_port_2 = uint16_t(control_port_0 + 2);


const char *Fabric_test::domain_name_verbs = ::getenv("DOMAIN_VERBS");
const char *Fabric_test::remote_host = ::getenv("SERVER");

std::ostream &describe_ep(std::ostream &o_, const component::IFabric_server_factory &e_)
{
	return o_ << "provider '" << e_.get_provider_name() << "' max_message_size " << e_.max_message_size();
}

namespace
{
	auto empty_object_json = common::json::serializer<common::json::dummy_writer>::object{};

	constexpr auto count_outer = 3U;
	constexpr auto count_inner = 3U;
	constexpr std::size_t memory_size = 4096;
	constexpr unsigned long iterations = 1000000;
	constexpr std::size_t msg_size = 1U << 6U;

	template <typename D>
		double double_seconds(D d_)
		{
			return std::chrono::duration<double>(d_).count();
		}

	std::pair<timeval, timeval> cpu_time()
	{
		struct rusage ru;
		{
			auto rc = ::getrusage(RUSAGE_SELF, &ru);
			EXPECT_EQ(rc, 0);
		}
		return { ru.ru_utime, ru.ru_stime };
	}

	std::chrono::microseconds usec(const timeval &start, const timeval &stop)
	{
		return
			( std::chrono::seconds(stop.tv_sec) + std::chrono::microseconds(stop.tv_usec) )
			-
			( std::chrono::seconds(start.tv_sec) + std::chrono::microseconds(start.tv_usec) )
		;
	}

	void ping_pong(const std::string &fabric_spec_, const char *const remote_host, std::uint16_t control_port_2_, unsigned thread_count_)
	{
		/* create object instance through factory */
		component::IBase * comp =
			component::load_component(
				"libcomponent-fabric.so",
				component::net_fabric_factory);
		ASSERT_TRUE(comp);

		auto factory = component::make_itf_ref(static_cast<component::IFabric_factory *>(comp->query_interface(component::IFabric_factory::iid())));
		auto fabric = std::shared_ptr<component::IFabric>(factory->make_fabric(fabric_spec_));
		auto control_port = std::uint16_t(control_port_2_);

		const std::size_t buffer_size = std::max(msg_size << 1U, memory_size);

		std::chrono::nanoseconds cpu_user;
		std::chrono::nanoseconds cpu_system;
		std::chrono::high_resolution_clock::duration t{};
		std::chrono::high_resolution_clock::duration start_stagger{};
		std::chrono::high_resolution_clock::duration stop_stagger{};
		std::uint64_t poll_count = 0U;
	
		if ( remote_host )
		{
			auto start_delay = std::chrono::seconds(3);
			/* allow time for the server to listen before the client restarts */
			std::this_thread::sleep_for(start_delay);

			std::cerr << "CLIENT begin port " << control_port << std::endl;

			using clock     = std::chrono::high_resolution_clock;

			auto start_time = clock::now();

			std::vector<std::future<pingpong_stat>> clients;
			/* In case the provider actually uses the remote keys which we provide, make them unique. */
			auto cpu_start = cpu_time();
			for ( auto remote_key_base = 0U; remote_key_base != thread_count_; ++remote_key_base )
			{
				clients.emplace_back(
					std::async(
						std::launch::async
						, [&fabric, remote_host, control_port, remote_key_base]
							{
							  auto id = static_cast<std::uint8_t>(remote_key_base);
							  pingpong_client client(*fabric, empty_object_json.str(), remote_host, control_port, buffer_size, remote_key_base, iterations, msg_size, id);
							  return client.time();
							}
					)
				);
			}
			auto start_max = std::chrono::high_resolution_clock::time_point::min();
			auto stop_max = std::chrono::high_resolution_clock::time_point::min();
			auto start_min = std::chrono::high_resolution_clock::time_point::max();
			auto stop_min = std::chrono::high_resolution_clock::time_point::max();

			for ( auto &f : clients )
			{
				auto r = f.get();
				start_min = std::min(start_min, r.start());
				start_max = std::max(start_max, r.start());
				stop_min = std::min(stop_min, r.stop());
				stop_max = std::max(stop_max, r.stop());
				poll_count += r.poll_count();
			}
			auto cpu_stop = cpu_time();
			cpu_user = usec(cpu_start.first, cpu_stop.first);
			cpu_system = usec(cpu_start.second, cpu_stop.second);
			std::cerr << "CLIENT end\n";
			__sync_synchronize();

			auto secs = std::chrono::duration<double>(clock::now() - start_time).count();
			double per_sec = double(iterations) / secs; 
			std::cerr << "Rate: " << per_sec << "/sec\n";
	
			t = stop_max - start_min;
			start_stagger = start_max - start_min;
			stop_stagger = stop_max - stop_min;
		}
		else
		{
			std::cerr << "SERVER (no remote) begin port " << control_port << std::endl;
	
			auto ep =
				std::unique_ptr<component::IFabric_server_factory>(
					fabric->open_server_factory(
						empty_object_json.str()
						, control_port)
				);
			EXPECT_LT(0U, ep->max_message_size());
			std::vector<std::future<pingpong_stat>> servers;
			/* In case the provider actually uses the remote keys which we provide, make them unique. */
			auto cpu_start = cpu_time();
			for ( auto remote_key_base = 0U; remote_key_base != thread_count_; ++remote_key_base )
			{
				servers.emplace_back(
					std::async(
						std::launch::async
						, [&ep, remote_key_base]
							{
								pingpong_server server(*ep, buffer_size, remote_key_base, iterations, msg_size);
								return server.time();
							}
					)
				);
			}

			auto start_max = std::chrono::high_resolution_clock::time_point::min();
			auto stop_max = std::chrono::high_resolution_clock::time_point::min();
			auto start_min = std::chrono::high_resolution_clock::time_point::max();
			auto stop_min = std::chrono::high_resolution_clock::time_point::max();
			for ( auto &f : servers )
			{
				auto r = f.get();
				start_min = std::min(start_min, r.start());
				start_max = std::max(start_max, r.start());
				stop_min = std::min(stop_min, r.stop());
				stop_max = std::max(stop_max, r.stop());
				poll_count += r.poll_count();
			}
			auto cpu_stop = cpu_time();
			cpu_user = usec(cpu_start.first, cpu_stop.first);
			cpu_system = usec(cpu_start.second, cpu_stop.second);
			std::cerr << "SERVER end\n";

			t = stop_max - start_min;
			start_stagger = start_max - start_min;
			stop_stagger = stop_max - stop_min;
		}
	
		auto iter = thread_count_ * iterations;
		auto iterf = static_cast<double>(iter);
		auto secs_inner = double_seconds(t);
		PINF("%zu byte PingPong, iterations/client: %lu clients: %u secs: %f start stagger: %f stop stagger: %f cpu_user: %f cpu_sys: %f Ops/Sec: %lu Polls/Op: %f"
			, msg_size
			, iterations
			, thread_count_
			, secs_inner
			, double_seconds(start_stagger)
			, double_seconds(stop_stagger)
			, double_seconds(cpu_user)
			, double_seconds(cpu_system)
			, static_cast<unsigned long>( iterf / secs_inner )
			, static_cast<double>(poll_count)/iterf
		);
	}

	void pingpong_single_server(const std::string &fabric_spec_, const char *const remote_host, std::uint16_t control_port_2, unsigned client_count_)
	{
		/* create object instance through factory */
		component::IBase * comp = component::load_component(
			"libcomponent-fabric.so",
			component::net_fabric_factory);
		ASSERT_TRUE(comp);

		auto factory = component::make_itf_ref(static_cast<component::IFabric_factory *>(comp->query_interface(component::IFabric_factory::iid())));
		auto fabric = std::shared_ptr<component::IFabric>(factory->make_fabric(fabric_spec_));
		auto control_port = std::uint16_t(control_port_2);

		const std::size_t buffer_size = std::max(msg_size << 1U, memory_size);
		std::uint64_t poll_count = 0U;

		std::chrono::nanoseconds cpu_user;
		std::chrono::nanoseconds cpu_system;
		std::chrono::high_resolution_clock::duration t{};
		std::chrono::high_resolution_clock::duration start_stagger{};
		std::chrono::high_resolution_clock::duration stop_stagger{};
	
		if ( remote_host )
		{
			auto start_delay = std::chrono::seconds(3);
			/* allow time for the server to listen before the client restarts */
			std::this_thread::sleep_for(start_delay);

			std::cerr << "CLIENT begin port " << control_port << std::endl;
			std::vector<std::future<pingpong_stat>> clients;
			/* In case the provider actually uses the remote keys which we provide, make them unique. */
			auto cpu_start = cpu_time();
			for ( auto remote_key_base = 0U; remote_key_base != client_count_; ++remote_key_base )
			{
				clients.emplace_back(
					std::async(
						std::launch::async
						, [&fabric, remote_host, control_port, remote_key_base]
							{
								auto id = static_cast<std::uint8_t>(remote_key_base);
								pingpong_client client(*fabric, empty_object_json.str(), remote_host, control_port, buffer_size, remote_key_base, iterations, msg_size, id);
								return client.time();
							}
					)
				);
			}
			auto start_max = std::chrono::high_resolution_clock::time_point::min();
			auto stop_max = std::chrono::high_resolution_clock::time_point::min();
			auto start_min = std::chrono::high_resolution_clock::time_point::max();
			auto stop_min = std::chrono::high_resolution_clock::time_point::max();

			for ( auto &f : clients )
			{
				auto r = f.get();
				start_min = std::min(start_min, r.start());
				start_max = std::max(start_max, r.start());
				stop_min = std::min(stop_min, r.stop());
				stop_max = std::max(stop_max, r.stop());
				poll_count += r.poll_count();
				std::cerr << "CLIENT end " << double_seconds(r.stop() - r.start()) << " sec" << std::endl;
			}
			auto cpu_stop = cpu_time();
			cpu_user = usec(cpu_start.first, cpu_stop.first);
			cpu_system = usec(cpu_start.second, cpu_stop.second);
			std::cerr << "CLIENT end" << std::endl;
			t = stop_max - start_min;
			start_stagger = start_max - start_min;
			stop_stagger = stop_max - stop_min;
		}
		else
		{
			std::cerr << "SERVER begin port " << control_port << std::endl;

			auto ep =
				std::unique_ptr<component::IFabric_server_factory>(
					fabric->open_server_factory(empty_object_json.str() , control_port)
				);
			EXPECT_LT(0U, ep->max_message_size());

			/* In case the provider actually uses the remote keys which we provide, make them unique. */
			auto remote_key_base = 0U;
			auto cpu_start = cpu_time();
			pingpong_server_n server(client_count_, *ep, buffer_size, remote_key_base, iterations, msg_size);
			auto f2 = server.time();
			auto cpu_stop = cpu_time();
			cpu_user = usec(cpu_start.first, cpu_stop.first);
			cpu_system = usec(cpu_start.second, cpu_stop.second);
			t = f2.stop() - f2.start();
			poll_count += f2.poll_count();

			std::cerr << "SERVER end" << std::endl;
		}

		auto iter = client_count_ * iterations;
		auto iterf = static_cast<double>(iter);
		auto secs_inner = double_seconds(t);

		PINF("%zu byte PingPong, iterations/client: %lu clients: %u secs: %f start stagger: %f stop stagger: %f cpu_user: %f cpu_sys: %f Ops/Sec: %lu Polls/Op: %f"
			, msg_size
			, iterations
			, client_count_
			, secs_inner
			, double_seconds(start_stagger)
			, double_seconds(stop_stagger)
			, double_seconds(cpu_user)
			, double_seconds(cpu_system)
			, static_cast<unsigned long>( iterf / secs_inner )
			, static_cast<double>(poll_count)/iterf
		);
	}

	TEST_F(Fabric_test, PingPong_1Threads)
	{
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 1U);
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 1U);
	}

	TEST_F(Fabric_test, PingPong_2Threads)
	{
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 2U);
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 2U);
	}

	TEST_F(Fabric_test, PingPong_4Threads)
	{
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 4U);
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 4U);
	}

	TEST_F(Fabric_test, PingPong_8Threads)
	{
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 8U);
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 8U);
	}

	TEST_F(Fabric_test, PingPong_16Threads)
	{
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 16U);
		ping_pong(fabric_spec("verbs"), remote_host, control_port_2, 16U);
	}

	TEST_F(Fabric_test, PingPong1Server_1Client)
	{
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 1U);
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 1U);
	}

	TEST_F(Fabric_test, PingPong1Server_2Clients)
	{
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 2U);
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 2U);
	}

	TEST_F(Fabric_test, PingPong1Server_4Clients)
	{
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 4U);
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 4U);
	}

	TEST_F(Fabric_test, PingPong1Server_8Clients)
	{
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 8U);
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 8U);
	}

	TEST_F(Fabric_test, PingPong1Server_16Clients)
	{
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 16U);
		pingpong_single_server(fabric_spec("verbs"), remote_host, control_port_2, 16U);
	}

} // namespace

int main(int argc, char **argv)
{
	/* server invocation:
	 *   fabric-test1
	 * client in invocation:
	 *   SERVER=10.0.0.91 fabric-test1
	 */

	std::cerr << "Domain/verbs is " << (Fabric_test::domain_name_verbs ? Fabric_test::domain_name_verbs : "unspecified") << "\n";

	::testing::InitGoogleTest(&argc, argv);
	auto r = RUN_ALL_TESTS();

	return r;
}
