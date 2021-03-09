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
 */

#ifndef _NUPM_DAX_MANAGER_ABSTRACT_H_
#define _NUPM_DAX_MANAGER_ABSTRACT_H_

#include "region_descriptor.h"
#include <common/string_view.h>
#include <cstddef>

namespace nupm
{
	struct dax_manager_abstract
	{
		using arena_id_t = unsigned;
		using string_view = common::string_view;

		virtual ~dax_manager_abstract() {}

  /**
   * Open a region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param out_length Out length of region in bytes
   *
   * @return backing file name (empty string if none);
   *   (pointer, length) pairs to the mapped memory, or empty vector
   *   if not found.
   *   Until fsdax supports extending a region, the vector will not be more
   *   than one element long.
   */
		virtual region_descriptor open_region(string_view id, arena_id_t arena_id) = 0;

  /**
   * Create a new region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param size Size of the region requested in bytes
   *
   * @return backing file name (empty string if none);
   *   Pointer to and size of mapped memory
   */
		virtual region_descriptor create_region(string_view id, arena_id_t arena_id, const size_t size) = 0;

  /**
   * Resize a region of memory
   *
   * @param id Unique identifier
   * @param arena_id Arena identifier
   * @param size Requested new size of the region requested in bytes
   *
   * The new size is just a request. The size may not change if, for example,
   * the underlying mechanism does not support resize. If the size does change,
   * it may change to something other than the requested size, due to rounding.
   * Returned values will not move existing mappings.
   *
   * Since open_region does not change state (perhaps it should be renamed
   * locate_region), it can be used to retrieve the new mapping of a resized region.
   *
   */
		virtual region_descriptor resize_region(string_view id, arena_id_t arena_id, size_t size) = 0;

  /**
   * Erase a previously allocated region
   *
   * @param id Unique region identifier
   * @param arena_id Arena identifier
   */
		virtual void erase_region(string_view id, arena_id_t arena_id) = 0;

  /**
   * Get the maximum "hole" size.
   *
   *
   * @return Size in bytes of max hole
   */
		virtual size_t get_max_available(arena_id_t arena_id) = 0;

  /**
   * Debugging information
   *
   * @param arena_id Arena identifier
   */
		virtual void debug_dump(arena_id_t arena_id) = 0;
#if 0
		virtual void register_range(const void *begin, std::size_t size) = 0;
		virtual void deregister_range(const void *begin, std::size_t size) = 0;
#endif
	};
}  // namespace nupm

#endif
