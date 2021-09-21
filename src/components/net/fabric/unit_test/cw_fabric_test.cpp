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

#include "cw_fabric_test.h"

#include <common/env.h>

#include <cstddef> /* size_t */
#include <cstdint> /* uint16_t */

const std::uint16_t fabric_fabric::control_port = common::env_value<std::uint16_t>("FABRIC_TEST_CONTROL_PORT", 47591);
const std::size_t fabric_fabric::data_size = common::env_value<std::size_t>("SIZE", 1U<<23);
const std::size_t fabric_fabric::memory_size = fabric_fabric::data_size + 100;
