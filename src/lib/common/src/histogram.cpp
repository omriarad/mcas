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

#include <common/histogram.h>

#include <numeric> /* accumulate */
#include <iostream> /* ostream */

common::hist_log2_out::hist_log2_out(const hist_log2 &log_, common::string_view units_)
	: hist_log2(log_)
	, units(units_)
{
}

auto common::hist_log2::count() const -> counter_type
{
	/* May or may not want a separate sample count, depending on how ofteh we reference it */
#if 0
	/* not using a separate cample count */
	return std::accumulate(hl_.begin(), hl_.end(), counter_type(0));
#else
	/* using a separate cample count */
	return _count;
#endif
}

common::hist_log2_out common::hist_log2::out(common::string_view units_)
{
	return hist_log2_out(*this, units_);
}

std::ostream & common::operator<<(std::ostream &os_, common::hist_log2_out hl_)
{
	using counter_type = common::hist_log2_out::counter_type;
	counter_type total = hl_.count();
	if ( total != 0 )
	{
		auto it = hl_.begin();
		for ( counter_type printed = 0; it != hl_.end() && printed != total; ++it )
		{
			auto i = *it;
			if ( i && ! printed )
			{
				os_ << "(" << (1U << (it - hl_.begin())) << " " << hl_.units << ") ";
			}
			if ( i || ( printed && printed != total ) )
			{
				os_ << i << " ";
				printed += i;
			}
		}
		os_ << "(" << (1U << (it - hl_.begin())) << " " << hl_.units << ")";
	}
	return os_;
}
