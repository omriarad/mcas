/*
   Copyright [2021] [IBM Corporation]
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


#ifndef MCAS_HSTORE_PERSIST_STRING_USE_H
#define MCAS_HSTORE_PERSIST_STRING_USE_H

/* Use of pmem_memcpy degrades performance unless we tell it what we are
 * using the string to hold: key or data. Keys must be copied with
 * "temporal store", as they will be read almost immediately. Data can be
 * copied with "nontemporal store", as it will not be re-read soon.
 */
enum class persist_string_use
{
	key,
	data,
	unknown
};

#endif
