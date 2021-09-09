/*
	 Copyright [2017-2019] [IBM Corporation]
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
#include "wait_poll.h"

#include <cassert>
#include <chrono> /* seconds */
#include <cstddef> /* size_t */

#if 0
namespace
{
	/* A callback which simply rejects (for requeue) any callback it comes across */
	component::IFabric_endpoint_connected::cb_acceptance reject(void *, ::status_t, std::uint64_t, std::size_t, void *)
	{
		return component::IFabric_endpoint_connected::cb_acceptance::DEFER;
	}
}
#endif

unsigned wait_poll(
	gsl::not_null<component::IFabric_endpoint_connected * > comm_
	, std::function<
		void(
			::fi_context2 *context
			, ::status_t
			, std::uint64_t completion_flags
			, std::size_t len
			, void *error_data
		)
	> cb_
)
{
	std::size_t ct = 0;
	unsigned poll_count = 0;
	while ( ct == 0 )
	{
		++poll_count;
		ct += comm_->poll_completions(cb_);
	}
	/* poll_completions does not always get a completion after wait_for_next_completion returns
	 * (does it perhaps return when a message begins to appear in the completion queue?)
	 * but it should not take more than two trips through the loop to get the completion.
	 *
	 * The sockets provider, though takes many more.
	 */
	assert(ct == 1);
	return poll_count;
}
