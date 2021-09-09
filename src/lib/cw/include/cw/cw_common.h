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
#ifndef CW_COMMON_H
#define CW_COMMON_H

#include <cstddef> /* size_t */

#include <api/fabric_itf.h> /*  fi_context2 */

#include <common/byte_span.h>
#include <common/histogram.h>
#include <common/moveable_ptr.h>
#include <common/types.h>

#include <boost/io/ios_state.hpp>
#include <gsl/pointers>

#include <sys/uio.h> /* iovec */

#include <chrono>
#include <cstddef> /* size_t */
#include <cstdint> /* uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <functional> /* function */
#include <iostream> /* cerr */
#include <vector>

namespace component
{
	struct IFabric_client;
	struct IFabric_endpoint_connected;
	struct IFabric_memory_control;
	struct IFabric_memory_region;
}
struct fi_context2;

#if 0
namespace cw
{
	struct memory
	{
		virtual ~memory() {}
		virtual char *data() = 0;
		virtual const char *data() const = 0;
		virtual std::size_t size() const = 0;
	};

	struct dram_memory
		: public memory
	{
	private:
		std::vector<char> _m;
	public:
		explicit dram_memory(std::size_t s)
			: _m(s)
		{}
		char *data() override { return _m.data(); }
		const char *data() const override { return _m.data(); }
		std::size_t size() const override { return _m.size(); }
	};

	struct pmem_memory
		: public memory
	{
		nupm::space_opened _space;
	private:
	public:
		explicit pmem_memory(nupm::space_opened &&);
	};
}
#else
#include "cw_memory.h"
#endif

namespace cw
{
	enum class quit_option : char
	{
		do_quit = 'q', /* on client_connected end, ask server to exit */
		do_not_quit = 'n', /* on client_connected end, ask server to end testcase */
		remain = 'r', /* on client_connected end, send nothing to server */
	};

	struct registration
	{
	private:
		common::moveable_ptr<component::IFabric_memory_control> _cnxn;
		component::IFabric_memory_region *_region;
		std::uint64_t _key;
		void * _desc;
	public:
		explicit registration();
		explicit registration(
			gsl::not_null<component::IFabric_memory_control *> cnxn_
			, common::const_byte_span span_
			, std::uint64_t key_
			, std::uint64_t flags_
		);
		registration(const registration &) = delete;
		registration &operator=(const registration &) = delete;
		registration(registration &&) noexcept = default;
		registration &operator=(registration &&) noexcept = default;
		~registration();

		std::uint64_t key() const { return _key; }
		void *desc() const { return _desc; }
	};

	struct registered_memory
	{
		using byte = common::byte;
	private:
		std::unique_ptr<memory> _memory;
		registration _registration;
	public:
		/*
		 * NOTE: if the memory remote key is used (that is, if the mr attributes do not include FI_PROV_KEY),
		 * the key must the unique among registered memories.
		 */
private:
		explicit registered_memory(gsl::not_null<component::IFabric_memory_control *> cnxn, std::size_t size, std::uint64_t remote_key)
			: registered_memory(cnxn, std::make_unique<dram_memory>(size), remote_key)
		{}
public:
		explicit registered_memory(gsl::not_null<component::IFabric_memory_control *> cnxn, std::unique_ptr<memory> &&m, std::uint64_t remote_key);
		explicit registered_memory();
		registered_memory(registered_memory &&rm) noexcept = default;
		registered_memory& operator=(registered_memory &&rm) noexcept = default;

		byte &at(std::size_t ix);
		byte &operator[](std::size_t ix) { return at(ix); }

		std::uint64_t key() const { return _registration.key(); }
		void *desc() const { return _registration.desc(); }
	};

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
	);

	struct remote_memory_accessor
		: fi_context2
	{
		remote_memory_accessor()
		{}
		void send_memory_info(component::IFabric_endpoint_connected &cnxn, registered_memory &rm);

		/* using rm as a buffer, send message */
		void send_msg(component::IFabric_endpoint_connected &cnxn, registered_memory &rm, const void *msg, std::size_t len);
	};

	using cvk_type =
		std::tuple<
			gsl::not_null<component::IFabric_client *>
			, std::uint64_t
			, std::uint64_t
		>;
	struct remote_memory_client_connected
		: public remote_memory_accessor
	{
	private:
		std::shared_ptr<registered_memory> _rm_out;
		std::shared_ptr<registered_memory> _rm_in;
		cvk_type _cnxn_and_vaddr_and_key;
		quit_option _quit_flag;
		::status_t _last_stat;

		status_t wait_complete();
		static void check_complete_static(void *t, void *ctxt, ::status_t stat, std::size_t len);
		void check_complete(::status_t stat, std::size_t len);
	public:
		remote_memory_client_connected(
			std::shared_ptr<registered_memory> rm_out
			, std::shared_ptr<registered_memory> rm_in
			, cvk_type cnxn_and_vaddr_and_key
			, quit_option quit_flag
		);
		~remote_memory_client_connected();

		registered_memory &rm_in() const { return *_rm_in; }
		registered_memory &rm_out() const { return *_rm_out; }
		component::IFabric_client &cnxn() const { return *std::get<0>(_cnxn_and_vaddr_and_key); }
		std::uint64_t vaddr() const { return std::get<1>(_cnxn_and_vaddr_and_key); }
		std::uint64_t key() const { return std::get<2>(_cnxn_and_vaddr_and_key); }
		std::tuple<uint64_t, uint64_t> set_vaddr_key(uint64_t vaddr, uint64_t key)
		{
			auto &v = std::get<1>(_cnxn_and_vaddr_and_key);
			auto &k = std::get<2>(_cnxn_and_vaddr_and_key);
			std::tuple<uint64_t, uint64_t> r(v, k);
			
			v = vaddr;
			k = key;

			return r;
		}
		std::tuple<uint64_t, uint64_t> set_vaddr_key(std::tuple<uint64_t, uint64_t> vk)
		{
			auto &v = std::get<1>(_cnxn_and_vaddr_and_key);
			auto &k = std::get<2>(_cnxn_and_vaddr_and_key);
			std::tuple<uint64_t, uint64_t> r(v, k);

			v = std::get<0>(vk);
			k = std::get<1>(vk);

			return r;
		}

		status_t write(common::string_view msg);
		status_t write_uninitialized(std::size_t s);
		status_t write_to_test(void *src, std::size_t sz, void *desc);
		status_t write_from_test(std::size_t sz, std::uint64_t vaddr, std::uint64_t key );
		status_t read_to_test(std::size_t sz, std::uint64_t vaddr, std::uint64_t key );
		void *write_from_test_local_addr() const;
		void *write_to_test_remote_addr() const;

		status_t read(std::size_t s);
		status_t read_verify(common::string_view msg_);
		std::size_t max_message_size() const;
#if 0
		{
			return cnxn().max_message_size();
		}
#endif
		void send_disconnect(component::IFabric_endpoint_comm &cnxn_, registered_memory &rm_, quit_option quit_flag_);
	};

	cvk_type get_rm(
		component::IFabric_client *cnxn
		, fi_context2 *expected_ctxt
		, common::byte *rm
	);

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
		ct()
			: common::hist_log2()
		{}

		void record(double t)
		{
			common::hist_log2::record(t*1e6);
		}
	};
}

#endif
