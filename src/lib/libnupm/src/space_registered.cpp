/*
   Copyright [2020-2021] [IBM Corporation]
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


/*
 * Authors:
 *
 */

#include <nupm/space_registered.h>

using nupm::space_registered;

space_registered::space_registered(
  const common::log_source &ls_
  , range_manager * rm_
  , common::fd_locked &&fd_
  , const string_view &name_
  , addr_t base_addr_
  , bool map_locked_
)
	: _pu(ls_, name_)
	, _or(ls_, rm_, std::move(fd_), base_addr_, map_locked_)
{
}

space_registered::space_registered(
  const common::log_source &ls_
  , nupm::range_manager * rm_
  , common::fd_locked &&fd_
  , const string_view &name_
  , const std::vector<byte_span> &mapping_
)
  : _pu(ls_, name_)
  , _or(ls_, rm_, std::move(fd_), mapping_)
{
}
