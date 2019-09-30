/////////////////////////////////////////////////////////////////////////////
// Copyright (c) Electronic Arts Inc. All rights reserved.
/////////////////////////////////////////////////////////////////////////////

#ifndef EASTL_INTERNAL_TRACED_H
#define EASTL_INTERNAL_TRACED_H

#include <EASTL/internal/config.h>

#include <EASTL/region_modifications.h>

#include <cstddef> /* size_t */
#include <cstring> /* memcpy/move/set */
#include <type_traits> /* remove_pointer_t */
#include <utility> /* move */

////////////////////////////////////////////////////////////////////////////////////////////
// Support for tracked writes.
////////////////////////////////////////////////////////////////////////////////////////////

namespace eastl
{
	template <typename Type, char ID>
		class tracked;

	class tracked_temp{};
	namespace tracking
	{
		static inline void
			track_raw(
				void *p
				, std::size_t s
				, char c = 'w' // w => raw
			) noexcept
		{

			nupm::region_tracker_add(p, s, c);
		}

		static auto memcpy(void *dst, const void *src, std::size_t ct)
		{
			auto r = ::memcpy(dst, src, ct);
			track_raw(dst, ct);
			return r;
		}

		static auto memmove(void *dst, const void *src, std::size_t ct)
		{
			auto r = ::memmove(dst, src, ct);
			track_raw(dst, ct);
			return r;
		}

		static auto memset(void *dst, char c, std::size_t ct)
		{
			auto r = ::memset(dst, c, ct);
			track_raw(dst, ct);
			return r;
		}
	}
	/*
	 * template for values whose access to memory must be tracked
	 */

	template <typename Type, char ID='?'>
		class tracked
		{
			/* Note: consider making type privately inherited if it is a class/struct,
			 * to accomodate empty classes
			 */
			Type _t;

			void track() noexcept
			{
				tracking::track_raw(&_t, sizeof _t, ID);
			}
		private:
			/* internal constructor */
			tracked(const Type &t_, tracked_temp) noexcept
				: _t(t_)
			{
			}
		public:
			/* "Rule of five" declarations */
			tracked(const Type &t_) noexcept
				: _t(t_)
			{
				track();
			}

			tracked(const tracked<Type, ID> &other_) noexcept
				: _t(other_._t)
			{
				track();
			}

			tracked(tracked<Type, ID> &&other_) noexcept
				: _t(std::move(other_._t))
			{
				track();
				/* the source may have been altered too */
				other_.track();
			}

			tracked &operator=(const Type &t_) noexcept
			{
				_t = t_;
				track();
				return *this;
			}

			tracked &operator=(const tracked<Type, ID> &other_) noexcept
			{
				_t = other_._t;
				track();
				return *this;
			}

			template <char I>
				tracked &operator=(const tracked<Type, I> &other_) noexcept
				{
					_t = other_._t;
					track();
					return *this;
				}

			~tracked() = default;

			/* zero-argument constructor */
			tracked() = default;
			/* Read access to the value */
			operator Type() const
			{
				return _t;
			}
			/* explicit read access */
			const Type &value() const
			{
				return _t;
			}
			/* explicit write access */
			Type &value()
			{
				track();
				return _t;
			}
			/* scalars */
			tracked<Type, ID> &operator++()
			{
				++_t;
				track();
				return *this;
			}
			template <typename U>
				tracked<Type, ID> &operator+=(const U &u)
				{
					_t += u;
					track();
					return *this;
				}
			template <typename U>
				tracked<Type, ID> &operator-=(const U &u)
				{
					_t -= u;
					track();
					return *this;
				}
			Type operator++(int)
			{
				auto t = _t;
				++_t;
				track();
				return t;
			}
			tracked<Type, ID> &operator--()
			{
				--_t;
				track();
				return *this;
			}
			Type operator--(int)
			{
				auto t = _t;
				--_t;
				track();
				return t;
			}

			/* pointers */
			std::remove_pointer_t<const Type> &operator*() const
			{
				return *_t;
			}

			std::remove_pointer_t<Type> &operator*()
			{
				/* presume that the non-const version is the target of a store */
				track();
				return *_t;
			}

			const Type &operator->() const
			{
				return _t;
			}

			Type &operator->()
			{
				/* presume that the non-const version is the target of a store */
				track();
				return _t;
			}

			template <typename U>
				tracked<U *, ID> ptr_cast() const
				{
					return tracked<U *, ID>(static_cast<U *>(_t), tracked_temp{});
				}

			template <typename U, char I>
				void swap(tracked<U, I> &b) noexcept
				{
					eastl::swap(_t, b._t);
					track();
					b.track();
				}

			template <typename U, char I>
				friend class eastl::tracked;

			void swap(tracked<Type, ID> &a, tracked<Type, ID> &b) noexcept;
		};

	template <typename Type, char ID>
		bool operator==(const tracked<Type, ID> &a, const tracked<Type, ID> &b)
		{
			return Type(a) == Type(b);
		}

	template <typename T, char ID>
		struct iterator_traits<tracked<T*, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef T*                                       pointer;
			typedef T&                                       reference;
		};

	template <typename T, char ID>
		struct iterator_traits<tracked<const T*, ID>>
		{
			typedef EASTL_ITC_NS::random_access_iterator_tag iterator_category;
			typedef T                                        value_type;
			typedef ptrdiff_t                                difference_type;
			typedef const T*                                 pointer;
			typedef const T&                                 reference;
		};

	template <typename Type, char ID>
	        inline void swap(tracked<Type, ID> &a, tracked<Type, ID> &b) noexcept
		{
			a.swap(b);
		}

} // namespace eastl

#endif // EASTL_INTERNAL_TRACED_H

