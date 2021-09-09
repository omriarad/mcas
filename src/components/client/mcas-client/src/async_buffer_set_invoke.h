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

#ifndef MCAS_CLIENT_ASYNC_BUFFER_SET_INVOKE_H
#define MCAS_CLIENT_ASYNC_BUFFER_SET_INVOKE_H

#include "async_buffer_set_t.h"

#include "connection.h"

#include <api/mcas_itf.h>
#include <common/delete_copy.h>
#include <vector>

struct async_buffer_set_invoke : public async_buffer_set_t
{
	std::vector<component::IMCAS::ADO_response> *out_ado_response;
	using Connection = mcas::client::Connection;

public:
	async_buffer_set_invoke(
		unsigned                          debug_level_,
		iob_ptr &&                        iobs_,
		iob_ptr &&                        iobr_,
		std::vector<component::IMCAS::ADO_response> *out_ado_response_
	)
		: async_buffer_set_t(debug_level_, std::move(iobs_), std::move(iobr_)),
			out_ado_response(out_ado_response_)
	{
	}
	DELETE_COPY(async_buffer_set_invoke);
	int              move_along(TM_ACTUAL Connection *c) override
	{
		TM_SCOPE(async_buffer_set_invoke)
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
			return c->receive_and_process_ado_response(iobr, *out_ado_response);
		}
		else {
			throw API_exception("invalid async handle, task already completed?");
		}
	}
};

#endif
