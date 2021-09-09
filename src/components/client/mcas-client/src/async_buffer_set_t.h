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

#ifndef MCAS_ASYNC_BUFFER_SET_H
#define MCAS_ASYNC_BUFFER_SET_H

#include "connection.h"
#include "iob_free.h"
#include "protocol.h"
#include "range.h"

#include <common/delete_copy.h>
#include <common/logging.h>
#include <common/utils.h>

#include <cstdlib>
#include <iostream>
#include <memory>

/**
 * @brief Used to track buffers for asynchronous invocations
 *
 * is an async handle.
 */
struct async_buffer_set_t : public component::IMCAS::Opaque_async_handle, protected common::log_source {
	using Connection = mcas::client::Connection;
	using iob_ptr = Connection::iob_ptr;
protected:
  iob_ptr        iobs;
  iob_ptr        iobr;

  async_buffer_set_t(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
    : component::IMCAS::Opaque_async_handle{},
      common::log_source(debug_level_),
      iobs(std::move(iobs_)),
      iobr(std::move(iobr_))
  {
    CPLOG(2, "%s iobs %p iobr %p"
          , __func__
          , common::p_fmt(&*iobs)
          , common::p_fmt(&*iobr)
          );
  }

  async_buffer_set_t()                           = delete;
  DELETE_COPY(async_buffer_set_t);

public:
  virtual ~async_buffer_set_t() {}
  virtual int move_along(TM_FORMAL Connection *c) = 0;
};

/* Nothing more than the two buffers. Used for async erase */
struct async_buffer_set_simple : public async_buffer_set_t {
  async_buffer_set_simple(unsigned debug_level_, iob_ptr &&iobs_, iob_ptr &&iobr_) noexcept
    : async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_))
  {
  }
  int move_along(TM_ACTUAL Connection *c) override
  {
    if (iobs) { /* check submission, clear and free on completion */
      if (c->test_completion(iobs->to_context()) == false) {
        return E_BUSY;
      }
      iobs.reset(nullptr);
    }

    if (iobr) { /* check recv, clear and free on completion */
      if (c->test_completion(iobr->to_context()) == false) {
        return E_BUSY;
      }

      const auto response_msg = c->msg_recv<const mcas::protocol::Message_IO_response>(&*iobr, "ASYNC PUT");

      auto status = response_msg->get_status();
      iobr.reset(nullptr);
      return status;
    }
    else {
      throw API_exception("invalid async handle, task already completed?");
    }
  }
};

#endif
