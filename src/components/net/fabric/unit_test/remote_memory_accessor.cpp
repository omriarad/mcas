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
#include "remote_memory_accessor.h"

#include "eyecatcher.h"
#include "registered_memory.h"
#include "wait_poll.h"
#include <api/fabric_itf.h> /* IFabric_endpoint_connected */
#include <common/errors.h> /* S_OK */
#include <common/logging.h> /* FLOG */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Weffc++"
#pragma GCC diagnostic ignored "-Wsign-compare"
#include <gtest/gtest.h>
#pragma GCC diagnostic pop

#include <sys/uio.h> /* iovec */
#include <cstdint> /* uint64_t */
#include <cstring> /* memcpy */
#include <exception>
#include <vector>


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
          EXPECT_EQ(ctxt_, this);
          EXPECT_EQ(stat_, S_OK);
        }
			, get_test_type()
    );
  }
  catch ( const std::exception &e )
  {
    FLOGM("exception {} {}", " exception ", e.what(), eyecatcher);
  }
}
