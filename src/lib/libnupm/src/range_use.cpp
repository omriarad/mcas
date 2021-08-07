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

#include "range_use.h"

#include "range_manager.h"

#include <boost/icl/split_interval_map.hpp>

#include <cinttypes>
#include <numeric> /* accumulate */
#include <sstream> /* otsringstream */
#include <stdexcept> /* runtiome_error */

using nupm::range_use;

std::vector<common::memory_mapped> range_use::address_coverage_check(std::vector<common::memory_mapped> &&iovm_)
{

  /* MCAS-86: we don't want this restriction, because different processes could be
     using the same virtual load address. */

	using AC = boost::icl::interval_set<byte *>;
	AC this_coverage;
	for ( const auto &e : iovm_ )
	{
		/* pointer_casts because the interval byte is, co far a char, not a gsl::byte or std::byte */
		boost::icl::discrete_interval<byte *> i = boost::icl::interval<byte *>::right_open(common::pointer_cast<byte>(::data(e)), common::pointer_cast<byte>(::data_end(e)));
		if ( _rm->interferes(i) )
		{
			std::ostringstream o;
			o << "range " << ::base(e) << ".." << ::end(e) << " overlaps existing mapped storage";
#if 0
			PLOG("%s: %s", __func__, o.str().c_str());
#endif
			throw std::runtime_error(o.str().c_str());
		}
		this_coverage.insert(i);
	}
	_rm->add_coverage(this_coverage);

	return std::move(iovm_);
}

range_use::range_use(range_manager *rm_, std::vector<common::memory_mapped> &&iovm_)
  : _rm(rm_)
  , _iovm(address_coverage_check(std::move(iovm_))) // std::vector<common::memory_mapped>())
{
	grow(std::move(iovm_));
}

range_use::~range_use()
{
	if ( bool(_rm) )
	{
		for ( const auto &e : _iovm )
		{
			/* pointer_casts because the interval byte is, so far, a char and not a gsl::byte or std::byte */
			auto i = boost::icl::interval<byte *>::right_open(common::pointer_cast<byte>(::data(e)), common::pointer_cast<byte>(::data_end(e)));
			_rm->remove_coverage(i);
		}
	}
}

void range_use::grow(std::vector<common::memory_mapped> &&iovv_)
{
	auto m = address_coverage_check(std::move(iovv_));
	std::move(m.begin(), m.end(), std::back_inserter(_iovm));
}

void range_use::shrink(std::size_t size_)
{
	while ( size_ != 0 )
	{
		auto &e = _iovm.back();
		if ( size_ < ::size(e) )
		{
			/* pointer_casts because the interval byte is, so far, a char and not a gsl::byte or std::byte */
			auto i = boost::icl::interval<byte *>::right_open(common::pointer_cast<byte>(::data_end(e)) - size_, common::pointer_cast<byte>(::data_end(e)));
			_rm->remove_coverage(i);
			_iovm.back().shrink_by(size_);
			size_ = 0;
		}
		else
		{
			/* pointer_casts because the interval byte is, so far, a char and not a gsl::byte or std::byte */
			auto i = boost::icl::interval<byte *>::right_open(common::pointer_cast<byte>(::data(e)), common::pointer_cast<byte>(::data_end(e)));
			_rm->remove_coverage(i);
			size_ -= ::size(e);
			_iovm.pop_back();
		}
	}
}

::off_t range_use::size() const
{
	return
		std::accumulate(
			_iovm.begin(), _iovm.end()
			, ::off_t(0)
			, [] (off_t a_, const common::memory_mapped & m_) { return a_ + ::size(m_); }
		);
}
