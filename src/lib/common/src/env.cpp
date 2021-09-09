/*
   Copyright [2013-2021] [IBM Corporation]
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

#include <common/env.h>

#include <cstdlib> /* getenv, strtod, strtoul */
#include <cstring> /* strlen */
#include <iostream> /* cout */
#include <limits>

namespace common
{
	template <>
		double env_value<double>(const char *const env_key, double dflt)
		{
			const char *env_str = std::getenv(env_key);
			if ( env_str )
			{
				char *endptr = nullptr;
				double env_value = std::strtod(env_str, &endptr);
				if ( endptr != env_str + std::strlen(env_str) )
				{
					std::cerr << "For key '" << env_key << "', value '" << env_str << "' is malformed, and ignored\n";
					goto fail;
				}
				dflt = env_value;
			}
		fail:
			return dflt;
		}

	unsigned long long env_selector<1>::g(const char *key_, unsigned long long dflt_, unsigned long long max_)
	{
		const char *env_str = std::getenv(key_);
		if ( env_str )
		{
			char *endptr = nullptr;
			auto env_value = std::strtoull(env_str, &endptr, 0);
			if ( endptr != env_str + std::strlen(env_str) )
			{
				std::cerr << "For key '" << key_ << "', value '" << env_str << "' is malformed, and ignored\n";
				goto fail;
			}
			if ( max_ < env_value )
			{
				std::cerr << "For key '" << key_ << "', value " << " exceeds " << max_ << ", and is ignored\n";
				goto fail;
			}
			dflt_ = env_value;
		}
	fail:
		return dflt_;
	}

	signed long long env_selector<2>::g(const char *const key_, signed long long dflt_, signed long long min_, signed long long mac_)
	{
		const char *env_str = std::getenv(key_);
		if ( env_str )
		{
			char *endptr = nullptr;
			auto env_value = std::strtoll(env_str, &endptr, 0);
			if ( endptr != env_str + std::strlen(env_str) )
			{
				std::cerr << "For key '" << key_ << "', value '" << env_str << "' is malformed, and ignored\n";
				goto fail;
			}
			if ( env_value < min_ )
			{
				std::cerr << "For key '" << key_ << "', value " << " is less than " << min_ << ", and is ignored\n";
				goto fail;
			}
			if ( mac_ < env_value )
			{
				std::cerr << "For key '" << key_ << "', value " << " exceeds " << mac_ << ", and is ignored\n";
				goto fail;
			}
			dflt_ = env_value;
		}
	fail:
		return dflt_;
	}

	template <>
		unsigned env_value<unsigned>(const char *const env_key, unsigned dflt)
		{
			return unsigned(env_value<unsigned long>(env_key, dflt));
		}

	template <>
		bool env_value<bool>(const char *const env_key, bool dflt)
		{
			return env_value(env_key, unsigned(dflt));
		}

	template <>
		const char *env_value<const char *>(const char *const env_key, const char *dflt)
		{
			const char *env_str = std::getenv(env_key);
			return env_str ? env_str : dflt;
		}

	template <>
		std::string env_value<std::string>(const char *const env_key, std::string dflt)
		{
			const char *env_str = std::getenv(env_key);
			return env_str ? std::string(env_str) : dflt;
		}
}
