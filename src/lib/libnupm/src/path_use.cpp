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

#include "path_use.h"

#include <common/logging.h>

#include <fcntl.h>
#include <algorithm> /* swap */
#include <mutex>
#include <set>
#include <sstream>
#include <string>
#include <stdexcept> /* range_error */

namespace
{
	std::set<std::string> nupm_dax_manager_mapped;
	std::mutex nupm_dax_manager_mapped_lock;
}

using nupm::path_use;

path_use::path_use(path_use &&other_) noexcept
  : common::log_source(other_)
  , _name()
{
  using std::swap;
  swap(_name, other_._name);
}

path_use::path_use(const common::log_source &ls_, const string_view &name_)
  : common::log_source(ls_)
  , _name(name_)
{
  std::lock_guard<std::mutex> g(nupm_dax_manager_mapped_lock);
  bool inserted = nupm_dax_manager_mapped.insert(std::string(name_)).second;
  if ( ! inserted )
  {
    std::ostringstream o;
    o << __func__ << ": instance already managing path (" << name_ << ")";
    throw std::range_error(o.str());
  }
  CPLOG(3, "%s (%p): name: %s", __func__, common::p_fmt(this), _name.c_str());
}

path_use::~path_use()
{
  if ( _name.size() )
  {
    std::lock_guard<std::mutex> g(nupm_dax_manager_mapped_lock);
    nupm_dax_manager_mapped.erase(_name);
    CPLOG(3, "%s: dax mgr instance: %s", __func__, _name.c_str());
  }
}
