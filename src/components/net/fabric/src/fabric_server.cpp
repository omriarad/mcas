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


#include "fabric_server.h"

#include "fabric_endpoint.h"
#include "fabric_enter_exit_trace.h"

Fabric_server::Fabric_server(
	gsl::not_null<component::IFabric_endpoint_unconnected_server *> aep_
)
	: Fabric_connection_server(aep_)
{}

Fabric_server::~Fabric_server()
{}

std::size_t Fabric_server::poll_completions(const component::IFabric_op_completer::complete_old &completion_callback)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions(completion_callback);
}

std::size_t Fabric_server::poll_completions(const component::IFabric_op_completer::complete_definite &completion_callback)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions(completion_callback);
}

std::size_t Fabric_server::poll_completions_tentative(const component::IFabric_op_completer::complete_tentative &completion_callback)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions_tentative(completion_callback);
}

std::size_t Fabric_server::poll_completions(const component::IFabric_op_completer::complete_param_definite &completion_callback, void *callback_param)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions(completion_callback, callback_param);
}

std::size_t Fabric_server::poll_completions_tentative(const component::IFabric_op_completer::complete_param_tentative &completion_callback, void *callback_param)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions_tentative(completion_callback, callback_param);
}

std::size_t Fabric_server::poll_completions(const component::IFabric_op_completer::complete_param_definite_ptr_noexcept completion_callback, void *callback_param)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions(completion_callback, callback_param);
}

std::size_t Fabric_server::poll_completions_tentative(const component::IFabric_op_completer::complete_param_tentative_ptr_noexcept completion_callback, void *callback_param)
{
	ENTER_EXIT_TRACE
	return aep()->poll_completions_tentative(completion_callback, callback_param);
}

std::size_t Fabric_server::stalled_completion_count()
{
	ENTER_EXIT_TRACE
	return aep()->stalled_completion_count();
}

void Fabric_server::wait_for_next_completion(unsigned polls_limit)
{
	ENTER_EXIT_TRACE
	return aep()->wait_for_next_completion(polls_limit);
}

void Fabric_server::wait_for_next_completion(std::chrono::milliseconds timeout)
{
	ENTER_EXIT_TRACE
	return aep()->wait_for_next_completion(timeout);
}

void Fabric_server::unblock_completions()
{
	ENTER_EXIT_TRACE
	return aep()->unblock_completions();
}

auto Fabric_server::register_memory(
	const_byte_span contig
	, std::uint64_t key
	, std::uint64_t flags
) -> memory_region_t
{
	ENTER_EXIT_TRACE
	return aep()->register_memory(contig, key, flags);
}

void Fabric_server::deregister_memory(
	const memory_region_t memory_region
)
{
	ENTER_EXIT_TRACE
	return aep()->deregister_memory(memory_region);
}

std::uint64_t Fabric_server::get_memory_remote_key(
	const memory_region_t memory_region
) const noexcept
{
	ENTER_EXIT_TRACE
	return aep()->get_memory_remote_key(memory_region);
}

void *Fabric_server::get_memory_descriptor(
	const memory_region_t memory_region
) const noexcept
{
	ENTER_EXIT_TRACE
	return aep()->get_memory_descriptor(memory_region);
}

auto Fabric_server::mr_covering(
	const const_byte_span contig_
) noexcept -> memory_region_t
{
	ENTER_EXIT_TRACE
	return aep()->mr_covering(contig_);
}

void Fabric_server::post_send(
	gsl::span<const ::iovec> buffers
	, void **desc
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_send(buffers, desc, context);
}

void Fabric_server::post_send(
	gsl::span<const ::iovec> buffers
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_send(buffers, context);
}

void Fabric_server::post_recv(
	gsl::span<const ::iovec> buffers
	, void **desc
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_recv(buffers, desc, context);
}

void Fabric_server::post_recv(
	gsl::span<const ::iovec> buffers
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_recv(buffers, context);
}

void Fabric_server::post_read(
	gsl::span<const ::iovec> buffers
	, void **desc
	, std::uint64_t remote_addr
	, std::uint64_t key
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_read(buffers, desc, remote_addr, key, context);
}

void Fabric_server::post_read(
	gsl::span<const ::iovec> buffers,
	std::uint64_t remote_addr,
	std::uint64_t key,
	context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_read(buffers, remote_addr, key, context);
}

void Fabric_server::post_write(
	gsl::span<const ::iovec> buffers
	, void **desc
	, std::uint64_t remote_addr
	, std::uint64_t key
	, context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_write(buffers, desc, remote_addr, key, context);
}

void Fabric_server::post_write(
	gsl::span<const ::iovec> buffers,
	std::uint64_t remote_addr,
	std::uint64_t key,
	context_t context
)
{
	ENTER_EXIT_TRACE
	return aep()->post_write(buffers, remote_addr, key, context);
}

void Fabric_server::inject_send(
	const void *buf, std::size_t len
)
{
	ENTER_EXIT_TRACE
	return aep()->inject_send(buf, len);
}
