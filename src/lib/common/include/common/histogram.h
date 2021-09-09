/*
   Copyright [2021 [IBM Corporation]
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

#ifndef MCAS_COMMON_HIST_LOG_H
#define MCAS_COMMON_HIST_LOG_H

#include <common/string_view.h>

#include <algorithm>  /* max */
#include <array> 
#include <cmath>  /* log2 */
#include <iosfwd> /* ostream */
#include <string>

namespace common
{
	struct hist_log2_out;

	struct hist_log2
		: private std::array<std::uint64_t, 64>
	{
		using counter_type = value_type;
	private:
		counter_type _count;
		using base = std::array<counter_type, 64>;

	public:
		hist_log2()
			: base{}
			, _count(0)
		{}

		void record(double t)
		{
			/* element 0 is [0..2) units */
			++at(unsigned(std::max(0.0, 0.0 < t ? std::log2(t) : 0)));
			++_count;
		}

		hist_log2_out out(common::string_view units);

		counter_type count() const;
		using base::begin;
		using base::end;
	};

	struct hist_log2_out
		: public hist_log2
	{
		std::string units;
		hist_log2_out(const hist_log2 &log_, common::string_view units_);
	};

	std::ostream & operator<<(std::ostream &os_, hist_log2_out hl);
}

#endif
