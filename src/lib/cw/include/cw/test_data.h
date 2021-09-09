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

#ifndef _CW_TEST_DATA_H
#define _CW_TEST_DATA_H

#include <algorithm> /* max */
#include <chrono>
#include <cstdint> /* uint64_t */

namespace cw
{
#if 0
	struct test_data
	{
		static constexpr std::uint64_t count() { return 10000; }
		static constexpr std::uint64_t size() { return 8ULL << 20; }
		static constexpr std::uint64_t memory_size() { return std::max(std::uint64_t(100), size()); }
		/* rest after no operation, for 4 milliseconds */
		static constexpr unsigned sleep_interval() { return 0; }
		static constexpr auto sleep_time() { return std::chrono::milliseconds(4); }
		static constexpr unsigned pre_ping_pong_interval() { return 1; }
		static constexpr unsigned post_ping_pong_interval() { return 1; }
		test_data(int) {}
	};
#endif
}

#endif
