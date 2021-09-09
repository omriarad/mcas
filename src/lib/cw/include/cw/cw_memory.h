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
#ifndef MCAS_CW_MEMORY_H_
#define MCAS_CW_MEMORY_H_

#include <common/byte_span.h>
#include <nupm/space_opened.h>
#include <cstddef> /* size_t */
#include <stdexcept>
#include <vector>

namespace cw
{
	struct memory
	{
		using byte = common::byte;
		virtual ~memory() {}
		virtual byte *data() = 0;
		virtual const byte *data() const = 0;
		virtual std::size_t size() const = 0;
	};

	struct dram_memory
		: public memory
	{
	private:
		std::vector<byte> _m;
	public:
		explicit dram_memory(std::size_t s)
			: _m(s)
		{}
		byte *data() override { return _m.data(); }
		const byte *data() const override { return _m.data(); }
		std::size_t size() const override { return _m.size(); }
	};

	struct pmem_memory
		: public memory
	{
		nupm::space_opened _space;
	private:
	public:
		explicit pmem_memory(nupm::space_opened && so)
			: _space(std::move(so))
		{
		}
		byte *data() override
		{
			if ( size() == 0 ) { throw std::runtime_error("accessing empty pmem_memory"); }
			return ::data(_space.range()[0]);
		}
		const byte *data() const override
		{
			if ( size() == 0 ) { throw std::runtime_error("accessing empty pmem_memory"); }
			return ::data(_space.range()[0]);
		}
		std::size_t size() const override { return ::size(_space.range()[0]); }
	};
}

#endif
