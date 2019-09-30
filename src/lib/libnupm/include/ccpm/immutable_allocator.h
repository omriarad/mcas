#ifndef __NUPM_CCPM_IMMUTABLE_ALLOCATOR_H__
#define __NUPM_CCPM_IMMUTABLE_ALLOCATOR_H__

#include <libpmem.h>
#include <common/utils.h>
#include <exception>
#include <atomic>

namespace ccpm
{
class Immutable_allocator_base
{
	static constexpr uint32_t MAGIC = 0xF1EEBBBB;
	const unsigned debug_level      = 3;

private:

  	struct Immutable_slab {
    	uint32_t              magic;
    	uint32_t              resvd;
    	byte *                slab_end;
    	Immutable_allocator_base * linked_slab;
    	std::atomic<byte *>   next_free;
  	} * _root __attribute__((packed));

  	static_assert(sizeof(Immutable_slab) == 32, "bad size Immutable_slab structure");
  	
public:

	/**
	 * @brief      Re-constructs allocator from existing memory
	 *
	 * @param      buffer  Memory region
	 */
	Immutable_allocator_base(void * buffer, size_t size) : _root(static_cast<Immutable_slab*>(buffer))
	{
		if(_root->magic != MAGIC) {
      if(size <= sizeof(struct Immutable_slab))
        throw std::bad_alloc();
      
      _root->magic = MAGIC;
      _root->slab_end = static_cast<byte*>(buffer) + size;
      _root->next_free = static_cast<byte*>(buffer) + sizeof(struct Immutable_slab);
      _root->linked_slab = nullptr;
      _root->resvd = 0;
      
      pmem_flush(_root, sizeof(*_root));
    }

		if(_root->next_free.load() > _root->slab_end)
			throw std::logic_error("next_free exceeds slab end");

    machine_checks();

		/* check that the memory is accessible */
		touch_pages(buffer, _root->slab_end - static_cast<byte*>(buffer));    
	}

  Immutable_allocator_base(Immutable_allocator_base& src) : _root(src._root) {}

  Immutable_allocator_base& operator=(const Immutable_allocator_base& src) {
    _root = src._root; return *this;
  }

  virtual ~Immutable_allocator_base() {
  }
  
	/**
	 * @brief      Allocate memory
	 *
	 * @param[in]  size  Size to allocate in bytes
	 *
	 * @return     Pointer to newly allocated region
	 */
	void * allocate(const size_t size)
	{
		if(size == 0)
			throw std::invalid_argument("size parameter cannnot be 0");

    byte * nptr;
    byte * updated_nptr;
    do {
      nptr = _root->next_free.load(std::memory_order_relaxed);
      updated_nptr = nptr + size;
      
      if(updated_nptr > _root->slab_end) /*< bounds check */
        throw std::bad_alloc();     
    }
    while(!_root->next_free.compare_exchange_weak(nptr,
                                                  updated_nptr,
                                                  std::memory_order_release,
                                                  std::memory_order_relaxed));

    pmem_flush(&_root->next_free, sizeof(_root->next_free));

		return nptr;
	}

  size_t remaining_size() const { return (_root->slab_end - _root->next_free); }

  virtual bool is_valid() const { return _root->magic == MAGIC; }

private:
  void machine_checks() {
    if(pmem_has_hw_drain() || pmem_has_auto_flush())
      throw std::logic_error("machine check failed");
  }
};



template<class T>
class Immutable_allocator
  : private Immutable_allocator_base,
    public std::allocator_traits<T>
{
public:
  Immutable_allocator(void * base, size_t size) noexcept
    : Immutable_allocator_base(base, size) {};

  T* allocate(std::size_t n) {
    return static_cast<T*>(Immutable_allocator_base::allocate(n * sizeof(T))); 
  }
  void deallocate(T*, std::size_t) noexcept {
    PWRN("deallocate on Immutable_allocator does nothing.");
  }
};

}

#endif // __NUPM_CCPM_IMMUTABLE_ALLOCATOR_H__
