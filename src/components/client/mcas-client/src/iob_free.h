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

#ifndef MCAS_CLIENT_IOB_FREE_H
#define MCAS_CLIENT_IOB_FREE_H

#include "connection.h"
/* The various embedded returns and throws suggest that the allocated
 * iobs should be automatically freed to avoid leaks.
 */

struct iob_free
{
private:
	mcas::client::Connection *_h;

public:
	iob_free(mcas::client::Connection *h_) : _h(h_) {}
	void operator()(mcas::client::Connection::buffer_t *iob) { _h->free_buffer(iob); }
};

#endif
