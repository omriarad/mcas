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


/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */

#ifndef _NUPM_CONFIG_T_H_
#define _NUPM_CONFIG_T_H_

#include <common/types.h> /* addr_t */
#include <string>

namespace nupm
{
  struct config_t {
    std::string path;
    addr_t addr;
    /* Through no fault of its own, config_t may begin life with no proper values */
    config_t() : path(), addr(0) {}
  };
}  // namespace nupm

#endif
