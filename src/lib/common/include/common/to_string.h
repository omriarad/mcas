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

#ifndef _MCAS_COMMON_TO_STRING_
#define _MCAS_COMMON_TO_STRING_

#include <sstream>

/* Convert stream arguments to a string */
namespace common
{
#if __cplusplus__ < 201703
	static inline void wr(std::ostream &)
	{
	}

	template<typename T, typename... Args>
		static void wr(std::ostream &o, const T & e, const Args & ... args)
		{
			o << e;
			wr(o, args...);
		}

	template <typename... Args>
		std::string to_string(const Args&... args)
		{
				std::ostringstream o;
				wr(o, args...);
				return o.str();
		}
#else
	template <typename... Args>
		std::string to_string(const Args&... args)
		{
			std::ostringstream s;
			(s << ... << args);
			return s.str();
		}
#endif
}

#if __cplusplus__ < 201703
static inline void common_wr(std::ostream &)
{
}

template<typename T, typename... Args>
	static void common_wr(std::ostream &o, const T & e, const Args & ... args)
	{
		o << e;
		common_wr(o, args...);
	}

template <typename... Args>
	std::string common_to_string(const Args&... args)
	{
			std::ostringstream o;
			common_wr(o, args...);
			return o.str();
	}
#else
template <typename... Args>
	std::string common_to_string(const Args&... args)
	{
		std::ostringstream s;
		(s << ... << args);
		return s.str();
	}
#endif

#endif
