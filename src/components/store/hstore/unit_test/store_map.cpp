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
#include "store_map.h"

#include <common/env.h>
#include <common/json.h>
#include <string>

const std::string store_map::impl_default = "hstore-cc";
const store_map::impl_map_t store_map::impl_map = {
  { "hstore-cc", { "hstore-cc", component::hstore_factory } }
  , { "hstore-mc", { "hstore-mc", component::hstore_factory } }
  , { "hstore", { "hstore", component::hstore_factory } }
};

namespace
{
  static auto store_env = common::env_value("STORE", "unspecified");
}

const store_map::impl_map_t::const_iterator store_map::impl_env_it = impl_map.find(store_env);

const store_map::impl_spec *const store_map::impl =
  & (
      impl_env_it == impl_map.end()
      ? impl_map.find(impl_default)
      : impl_env_it
    )->second
  ;

namespace c_json = common::json;
using json = c_json::serializer<c_json::dummy_writer>;

const auto devdax_location =
  json::array
  ( json::object
    ( json::member("path", "/dev/dax0.0")
    , json::member("addr", 0x9000000000)
    )
  );

const auto fsdax_location =
  json::array
  ( json::object
    ( json::member("path", "/mnt/pmem1")
    , json::member("addr", 0x9000000000)
    )
  );

  /* example of a STORE_LOCATION string:
   *   [{"path": "/mnt/pmem1", "addr": 618475290624}]
   */
const std::string store_map::location =
  common::env_value(
    "STORE_LOCATION"
    /* if USE_ODP was set, default is fsdax location, else default is devdax location */
    , common::env_value("USE_ODP", true) ? fsdax_location.str() : devdax_location.str()
  );
