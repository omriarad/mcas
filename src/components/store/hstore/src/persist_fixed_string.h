/*
   Copyright [2017-2019] [IBM Corporation]
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


#ifndef COMANCHE_HSTORE_PERSIST_FIXED_STRING_H
#define COMANCHE_HSTORE_PERSIST_FIXED_STRING_H

#include "persistent.h"

#include <algorithm>
#include <array>
#include <cstddef> /* size_t */
#include <string> /* to_string */
#include <tuple>

template <typename T, std::size_t SmallSize, typename Allocator>
	class persist_fixed_string;

template <typename T, std::size_t SmallSize, typename Allocator>
	union rep;

class fixed_string_access
{
	fixed_string_access() {}
public:
	template <typename T, std::size_t SmallSize, typename Allocator>
		friend union rep;
};

/*
 * - fixed_string
 * - ref_count: because the object may be referenced twice as the table expands
 * - size: length of data (data immediately follows the fixed_string object)
 */
template <typename T>
	class fixed_string
	{
		unsigned _ref_count;
		unsigned _alignment;
		uint64_t _size;
		uint64_t size() const { return _size; }
		/* offset to data, for a particular alignment */
		std::size_t front_pad() const noexcept { return data_offset() - sizeof *this; }
		std::size_t front_skip_element_count() const
		{
			return front_skip_element_count(_alignment);
		}
		static std::size_t data_offset(std::size_t alignment_) noexcept { return front_skip_element_count(alignment_) * sizeof(T); }

		template <typename I>
			static auto gcd(I a, I b) -> I
			{ /* Euclid's algorithm */
				while (true)
				{
					if (a == I()) return b;
					b %= a;
					if (b == I()) return a;
					a %= b;
				}
			}
		template <typename I>
			static auto lcm(I a, I b) -> I
			{
				auto g = gcd(a,b);
				return g ? (a/g * b) : I();
			}
		static std::size_t front_skip_element_count(std::size_t alignment_) noexcept
		{
			/* the offset in bytes must be a multiple of both alignment (to align the data)
			 * and sizeof(T) (since the returned unit is sizeof(T) bytes
			 */
			auto s = lcm(alignment_, sizeof(T));
			auto b = sizeof(fixed_string<T>) + s - 1; /* maximum required size in bytes */
			b = b / s * s; /* round down to a multiple of s */
			return b / sizeof(T);
		}
		std::size_t data_offset() const noexcept { return data_offset(_alignment); }
		std::size_t alloc_element_count() const
		{
			return front_skip_element_count() + size();
		}
	public:
		using access = fixed_string_access;
		template <typename IT>
			fixed_string(IT first_, IT last_, std::size_t pad_, std::size_t alignment_, access a_)
				: fixed_string(static_cast<std::size_t>(last_-first_ + pad_), alignment_, a_)
			{
				/* for small lengths we copy */
				/* fill for alignment, returning address of first aligned byte */
				const auto c0 = std::fill_n(static_cast<char *>(static_cast<void *>(this+1)), front_pad(), 0);
				/* first aligned element starts at first aligned byte */
				const auto e0 = static_cast<T *>(static_cast<void *>(c0));
				std::fill_n(
					std::copy(first_, last_, e0)
					, pad_
					, T()
				);
			}

		fixed_string(std::size_t data_len_, std::size_t alignment_, access)
			: _ref_count(1U)
			, _alignment(unsigned(alignment_))
			, _size(data_len_)
		{
			if ( _alignment != alignment_ )
			{
				throw std::domain_error("object alignment too large; probably exceeds 2^31");
			}
		}
	public:
		template <typename Allocator>
			void persist_this(const Allocator &al_)
			{
				al_.persist(this, sizeof *this + alloc_element_count() * sizeof(T));
			}
		uint64_t size(access) const noexcept { return size(); }
		uint64_t alignment(access) const noexcept { return _alignment; }
		unsigned inc_ref(access, int, const char *) noexcept { return _ref_count++; }
		unsigned dec_ref(access, int, const char *) noexcept { return --_ref_count; }
		unsigned ref_count(access) noexcept { return _ref_count; }
		const T *data(access) const
		{
			auto c0 = static_cast<const char *>(static_cast<const void *>(this)) + data_offset();
			auto e0 = static_cast<const T *>(static_cast<const void *>(c0));
			return e0;
		}

		T *data(access)
		{
			auto c0 = static_cast<char *>(static_cast<void *>(this)) + data_offset();
			auto e0 = static_cast<T *>(static_cast<void *>(c0));
			return e0;
		}

		static std::size_t front_skip_element_count(std::size_t alignment_, access)
		{
			return front_skip_element_count(alignment_);
		}

		std::size_t alloc_element_count(access) const
		{
			return alloc_element_count();
		}
	};

template <typename T, std::size_t SmallSize, typename Allocator>
	union rep
	{
		using element_type = fixed_string<T>;
		using allocator_type = typename Allocator::template rebind<element_type>::other;
		using allocator_char_type = typename allocator_type::template rebind<char>::other;
		using allocator_void_type = typename allocator_type::template rebind<void>::other;
		using ptr_t = persistent_t<typename allocator_type::pointer>;
		using access = fixed_string_access;

		struct small_t
		{
			std::array<char, SmallSize> value;
		private:
			bool _is_small : 1; /* discriminant */
			unsigned int _size : 7;
		public:
			/* note: as of C++17, can use std::clamp */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
			small_t(std::size_t size_, std::size_t alignment_)
				: value{}
				, _is_small(size_ <= sizeof value && alignment_ <= alignof(ptr_t))
				, _size(size_ <= sizeof value ? size_ : 0)
			{
			}
#pragma GCC diagnostic pop
			small_t(small_t &other) = default;
			bool is_small() const { return _is_small; }
			void set_small(bool s) { _is_small = s; }
			unsigned int size() const { return _size; }
		} small;

		struct large_t
			: public allocator_char_type
		{
			allocator_char_type &al() { return static_cast<allocator_char_type &>(*this); }
			const allocator_char_type &al() const { return static_cast<const allocator_char_type &>(*this); }
			ptr_t ptr;
		} large;

		static_assert(
			sizeof large <= sizeof small.value
			, "large_t overlays with small.size"
		);

		/* ERROR: caller needs to to persist */
		rep()
			: small(0, alignof(small_t))
		{
			large.ptr = nullptr;
		}

		template <typename IT, typename AL>
			rep(
				IT first_
				, IT last_
				, std::size_t fill_len_
				, std::size_t alignment_
				, AL al_
			)
				: small(static_cast<std::size_t>(last_ - first_ + fill_len_) * sizeof(T), alignment_)
			{
				if ( is_small() )
				{
					std::fill_n(
						std::copy(
							first_
							, last_
							, static_cast<T *>(static_cast<void *>(&small.value[0]))
						)
						, fill_len_
						, T()
					);
				}
				else
				{
					auto data_size =
						static_cast<std::size_t>(last_ - first_ + fill_len_) * sizeof(T);
					using local_allocator_char_type = typename AL::template rebind<char>::other;
					new (&large.al()) allocator_char_type(al_);
					new (&large.ptr)
						ptr_t(
							static_cast<typename allocator_type::pointer>(
								typename allocator_void_type::pointer(
									local_allocator_char_type(al_).allocate(element_type::front_skip_element_count(alignment_, access{}) + data_size, alignment_)
								)
							)
						);
					new (&*large.ptr) element_type(first_, last_, fill_len_, alignment_, access{});
					large.ptr->persist_this(al_);
				}
			}

		template <typename AL>
			rep(
				std::size_t data_len_
				, std::size_t alignment_
				, AL al_
			)
				: small(data_len_ * sizeof(T), alignment_)
			{
				/* large data sizes: data not copied, but space is reserved for RDMA */
				if ( is_small() )
				{
				}
				else
				{
					auto data_size = data_len_ * sizeof(T);
					new (&large.al()) allocator_char_type(al_);
					new (&large.ptr)
						ptr_t(
								static_cast<typename allocator_type::pointer>(
								typename allocator_void_type::pointer(
									al_.allocate(element_type::front_skip_element_count(alignment_, access{}) + data_size, alignment_)
								)
							)
						);
					new (&*large.ptr) element_type(data_size, alignment_, access{});
				}
			}

		rep(const rep &other)
			: small(other.size(), alignof(small_t))
		{
			if ( is_small() )
			{
				small = other.small;
			}
			else
			{
				small = other.small;
				new (&large.al()) allocator_char_type(other.large.al());
				new (&large.ptr) ptr_t(other.large.ptr);
				if ( large.ptr )
				{
					large.ptr->inc_ref(access{}, __LINE__, "ctor &");
				}
			}
		}

		rep(rep &&other)
			: small(other.small)
		{
			if ( this != &other )
			{
				if ( ! is_small() )
				{
					new (&large.al()) allocator_type(other.large.al());
					new (&large.ptr) ptr_t(other.large.ptr);
					other.large.ptr = typename allocator_type::pointer{};
				}
			}
		}

		/* Note: To handle "issue 41" updates, this operation must be restartable - must not alter "other" until this is persisted.
		 */
		rep &operator=(const rep &other)
		{
			if ( is_small() )
			{
				if ( other.is_small() )
				{
					/* small <- small */
					small = other.small;
				}
				else
				{
					/* small <- large */
					new (&large.al()) allocator_type(other.large.al());
					new (&large.ptr) ptr_t(other.large.ptr);
					large.ptr->inc_ref(access{}, __LINE__, "=&");
					small.set_small(false); /* "large" kind */
				}
			}
			else
			{
				/* large <- ? */
				if ( large.ptr && large.ptr->ref_count(access{}) != 0 && large.ptr->dec_ref(access{}, __LINE__, "=&") == 0 )
				{
					auto sz = large.ptr->alloc_element_count(access{});
					auto alignment = large.ptr->alignment(access{});
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), alignment, sz);
				}
				large.al().~allocator_char_type();

				if ( other.is_small() )
				{
					/* large <- small */
					small = other.small;
				}
				else
				{
					/* large <- large */
					large.ptr = other.large.ptr;
					large.ptr->inc_ref(access{}, __LINE__, "=&");
					new (&large.al()) allocator_type(other.large.al());
				}
			}
			return *this;
		}

		rep &operator=(rep &&other)
		{
			if ( is_small() )
			{
				if ( other.is_small() )
				{
					/* small <- small */
					small = other.small;
				}
				else
				{
					/* small <- large */
					new (&large.al()) allocator_type(other.large.al());
					new (&large.ptr) ptr_t(other.large.ptr);
					small.set_small(false); /* "large" flag */
					other.large.ptr = ptr_t();
				}
			}
			else
			{
				/* large <- ? */
				if ( large.ptr && large.ptr->ref_count(access{}) != 0 && large.ptr->dec_ref(access{}, __LINE__, "=&&") == 0 )
				{
					auto sz = large.ptr->alloc_element_count(access());
					auto alignment = large.ptr->alignment(access());
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), sz, alignment);
				}
				large.al().~allocator_char_type();

				if ( other.is_small() )
				{
					/* large <- small */
					small = other.small;
				}
				else
				{
					/* large <- large */
					large.ptr = other.large.ptr;
					new (&large.al()) allocator_type(other.large.al());
					other.large.ptr = ptr_t();
				}
			}
			return *this;
		}

		~rep()
		{
			if ( is_small() )
			{
			}
			else
			{
				if ( large.ptr && large.ptr->dec_ref(access{}, __LINE__, "~") == 0 )
				{
					auto sz = large.ptr->alloc_element_count(access());
					auto alignment = large.ptr->alignment(access());
					large.ptr->~element_type();
					large.al().deallocate(static_cast<typename allocator_char_type::pointer>(static_cast<typename allocator_void_type::pointer>(large.ptr)), sz, alignment);
				}
				large.al().~allocator_char_type();
			}
		}


		void deconstitute() const
		{
			if ( ! is_small() )
			{
				/* used only by the table_base destructor, at which time
				 * the reference count should be 1. There is not much point
				 * in decreasing the reference count except to mirror
				 * reconstitute.
				 */
				if ( large.ptr->dec_ref(access(), __LINE__, "deconstitute") == 0 )
				{
					large.al().~allocator_char_type();
				}
			}
		}
		template <typename AL>
			void reconstitute(AL al_) const
			{
				using reallocator_char_type = typename AL::template rebind<char>::other;
				if ( ! is_small() )
				{
					new (&const_cast<rep *>(this)->large.al()) allocator_char_type(al_);
					auto alr = reallocator_char_type(al_);
					if ( alr.is_reconstituted(large.ptr) )
					{
						/* The data has already been reconstituted. Increase the reference count. */
						large.ptr->inc_ref(access(), __LINE__, "reconstitute");
					}
					else
					{
						/* The data is not yet reconstituted. Reconstitute it.
						 * Although the original may have had a refcount
						 * greater than one, we have not yet seen the
						 * second reference, so the recount must be set to one.
						 */
						alr.reconstitute(large.ptr->alloc_element_count(access{}) * sizeof(T), large.ptr);
						new (large.ptr) element_type( size(), large.ptr->alignment(access{}), access{} );
					}
				}
			}

		bool is_small() const
		{
			return small.is_small();
		}

		std::size_t size() const
		{
			return is_small() ? small.size() : large.ptr->size(access{});
		}

		const T *data() const
		{
			if ( is_small() )
			{
				return static_cast<const T *>(&small.value[0]);
			}
			return large.ptr->data(access{});
		}

		T *data()
		{
			if ( is_small() )
			{
				return static_cast<T *>(&small.value[0]);
			}
			return large.ptr->data(access{});
		}
	};

template <typename T, std::size_t SmallSize, typename Allocator>
	class persist_fixed_string
	{
		using access = fixed_string_access;
		using element_type = fixed_string<T>;
		using EA = typename Allocator::template rebind<element_type>::other;
		using ptr_t = persistent_t<typename EA::pointer>;
		/* "rep" is most of persist_fixed_string; it is conceptually its base class
		 * It does not directly replace persist_fixed_string only to preserve the
		 * declaration of persist_fixed_string as a class, not a union
		 */
		rep<T, SmallSize, Allocator> _rep;
		/* NOTE: allocating the data string adjacent to the header of a fixed_string
		 * precludes use of a standard allocator
		 */

	public:
        static constexpr std::size_t default_alignment = 8;
		using allocator_type = Allocator;
		template <typename U>
			using rebind = persist_fixed_string<U, SmallSize, Allocator>;

		persist_fixed_string()
			: _rep()
		{
		}

		template <typename IT, typename AL>
			persist_fixed_string(
				IT first_
				, IT last_
				, std::size_t fill_len_
				, std::size_t alignment_
				, AL al_
			)
				: _rep(first_, last_, fill_len_, alignment_, al_)
			{
			}

		template <typename IT, typename AL>
			persist_fixed_string(
				IT first_
				, IT last_
				, AL al_
			)
				: persist_fixed_string(first_, last_, 0U, default_alignment, al_)
			{
			}

		template <typename AL>
			persist_fixed_string(
				std::size_t data_len_
				, AL al_
			)
				: _rep(data_len_, default_alignment, al_)
			{
			}

		persist_fixed_string(const persist_fixed_string &other)
			: _rep(other._rep)
		{
		}

		persist_fixed_string(persist_fixed_string &&other) = default;

		persist_fixed_string &operator=(const persist_fixed_string &other) = default;
		persist_fixed_string &operator=(persist_fixed_string &&other) = default;

		~persist_fixed_string()
		{
		}

		std::size_t size() const { return _rep.size(); }

		const T *data() const { return _rep.data(); }

		T *data() { return _rep.data(); }

		void deconstitute() const { return _rep.deconstitute(); }
		template <typename AL>
			void reconstitute(AL al_) const { return _rep.reconstitute(al_); }
	};

template <typename T, std::size_t SmallSize, typename Allocator>
	bool operator==(
		const persist_fixed_string<T, SmallSize, Allocator> &a
		, const persist_fixed_string<T, SmallSize, Allocator> &b
	)
	{
		return
			a.size() == b.size()
			&&
			std::equal(a.data(), a.data() + a.size(), b.data())
		;
	}

#endif
