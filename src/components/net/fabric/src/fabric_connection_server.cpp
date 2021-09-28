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


#include "fabric_connection_server.h"

#include "fabric_check.h" /* CHECK_FI_ERR */
#include "fabric_endpoint.h"
#include "fabric_enter_exit_trace.h"
#include "fd_control.h"

#include "rdma-fi_cm.h" /* fi_accept */

#include <cstdint> /* size_t */
#include <exception>
#include <memory> /* unique_ptr */

Fabric_connection_server::Fabric_connection_server(
  component::IFabric_endpoint_unconnected_server *aep_
)
  : fabric_connection(aep_, fabric_types::addr_ep_t{})
{
  if ( aep()->ep_info().ep_attr->type == FI_EP_MSG )
  {
    std::size_t paramlen = 0;
    auto param = nullptr;
    CHECK_FI_ERR(::fi_accept(&aep()->ep(), param, paramlen));
  }
}

/* The server does not need to do anything to solicit an event,
 * as the server_factory continuously reads the server's event queue
 */
void Fabric_connection_server::solicit_event() const
{
}

void Fabric_connection_server::wait_event() const
{
}

std::size_t Fabric_connection_server::max_message_size() const noexcept
{
	ENTER_EXIT_TRACE1
	return aep()->ep_info().ep_attr->max_msg_size;
}

std::size_t Fabric_connection_server::max_inject_size() const noexcept
{
	ENTER_EXIT_TRACE_N
	return aep()->ep_info().tx_attr->inject_size;
}
