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



/*
 * Authors:
 *
 * Daniel G. Waddington (daniel.waddington@ibm.com)
 *
 */
#include "uipc_shared_memory.h"

#include <common/exceptions.h>
#include <sys/ioctl.h>
#include <sys/mman.h>

#include <unistd.h>
#include <cassert>
#include <cstring>
#include <string>

#define FIFO_DIRECTORY "/tmp"

namespace Core
{
namespace UIPC
{
struct addr_size_pair {
  addr_t addr;
  size_t size;
  void *ptr() const { return reinterpret_cast<void *>(addr); }
};

/**
 * Shared_memory class
 *
 * @param name
 * @param n_pages
 */

Shared_memory::Shared_memory(const std::string &name, size_t n_pages)
    : _master(true)
    , _fifo_names()
    , _name(name)
    , _mapped_pages{
        (
          (void)(option_DEBUG && (PLOG("try to negotiate_shared_memory(%s): master", name.c_str()), true)),
          negotiate_addr_create(
            std::string(FIFO_DIRECTORY) + "/fifo." + name
            , n_pages * PAGE_SIZE
          )
        ),
        n_pages
      } {

  assert(_mapped_pages._vaddr);

  if (option_DEBUG)
    PLOG("open_shared_memory(%s): master", name.c_str());
  open_shared_memory(name, true);
}

Shared_memory::Shared_memory(const std::string &name)
    : _master(false)
    , _fifo_names()
    , _name(name)
    , _mapped_pages(
        (
          (void)(option_DEBUG && (PLOG("try to negotiate_shared_memory(%s): slave", name.c_str()), true)),
          negotiate_addr_connect(std::string(FIFO_DIRECTORY) + "/fifo." + name)
        )
      ) {
  assert(_mapped_pages._vaddr);
  if (option_DEBUG)
    PLOG("open_shared_memory(%s): slave", name.c_str());

  open_shared_memory(name, false);
}

Shared_memory::~Shared_memory() noexcept(false) {
  if(option_DEBUG)
    PLOG("unmapping shared memory: %p", _mapped_pages._vaddr);

  if (::munmap(_mapped_pages._vaddr, _mapped_pages._size_in_pages * PAGE_SIZE) != 0)
    throw General_exception("unmap failed");

  if (_master) {
    int rc = shm_unlink(_name.c_str());
    if (rc != 0)
      throw General_exception("shared memory failed to unlink (%s)",
                              _name.c_str());

    for (auto& n : _fifo_names) {
      if(option_DEBUG)
        PLOG("removing fifo (%s)", n.c_str());
      rc = unlink(n.c_str());
      if (rc != 0)
        throw General_exception("shared memory failed to remove fifo (%s)",
                                n.c_str());
    }
  }
}

void* Shared_memory::mapped_pages::get_addr(size_t offset) {
  if (offset >> (_size_in_pages * PAGE_SIZE))
    throw API_exception("invalid offset parameter");

  return static_cast<char *>(_vaddr) + offset;
}

void* Shared_memory::get_addr(size_t offset) {
  return _mapped_pages.get_addr(offset);
}

void Shared_memory::open_shared_memory(const std::string &name, bool master) {
  if(option_DEBUG)
    PLOG("open shared memory:%s %d", name.c_str(), master);

  umask(0);
  int fd = -1;

  if (master) {
    fd = ::shm_open(name.c_str(), O_CREAT | O_TRUNC | O_RDWR,
                  S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP);
  }
  else {
    while (fd == -1) {
      fd = ::shm_open(name.c_str(), O_RDWR,
                    S_IRUSR | S_IWUSR | S_IROTH | S_IWOTH | S_IRGRP | S_IWGRP);
      usleep(100000);
    }
  }

  if (fd == -1)
    throw Constructor_exception("shm_open failed to open/create %s",
                                name.c_str());

  if (::ftruncate(fd, _mapped_pages._size_in_pages * PAGE_SIZE))
    throw General_exception("unable to allocate shared memory IPC");

  void* ptr = ::mmap(_mapped_pages._vaddr, _mapped_pages._size_in_pages * PAGE_SIZE, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, fd, 0);
  if (ptr != _mapped_pages._vaddr)
    throw Constructor_exception("mmap failed in Shared_memory");

  if(master)
    std::memset(ptr, 0xbb, _mapped_pages._size_in_pages * PAGE_SIZE); /* important to do this only on master side */

  ::close(fd);
}

static addr_t VADDR_BASE = 0x8000000000;

static void wait_for_read(int fd, size_t s) {
  assert(s > 0);

  int count = 0;
  do {
    ::ioctl(fd, FIONREAD, &count);
  } while (unsigned(count) < s);
}

void* Shared_memory::negotiate_addr_create(const std::string &name,
                                           size_t size_in_bytes) {
  /* create FIFO - used to negotiate memory address */
  umask(0);

  std::string name_s2c = name + ".s2c";
  std::string name_c2s = name + ".c2s";

  if (option_DEBUG) {
    PLOG("mkfifo %s", name_s2c.c_str());
    PLOG("mkfifo %s", name_c2s.c_str());
  }

  assert(_master);

  unlink(name_c2s.c_str());
  unlink(name_s2c.c_str());

  if (mkfifo(name_c2s.c_str(), 0666) || mkfifo(name_s2c.c_str(), 0666)) {
    perror("mkfifo:");
    throw General_exception("mkfifo failed in negotiate_addr_create");
  }

  int fd_s2c = ::open(name_s2c.c_str(), O_WRONLY);
  int fd_c2s = ::open(name_c2s.c_str(), O_RDONLY);

  assert(fd_c2s >= 0 && fd_s2c >= 0);

  if(option_DEBUG) {
    PLOG("saving fifo name: %s", name_c2s.c_str());
    PLOG("saving fifo name: %s", name_s2c.c_str());
  }
  _fifo_names.push_back(name_c2s);
  _fifo_names.push_back(name_s2c);

  addr_t vaddr = VADDR_BASE;
  void* ptr = nullptr;

  do {
    ptr = ::mmap(reinterpret_cast<void*>(vaddr), size_in_bytes, PROT_NONE,
               MAP_SHARED | MAP_ANONYMOUS, 0, 0);

    if ((ptr == reinterpret_cast<void*>(-1)) || (ptr != reinterpret_cast<void*>(vaddr))) {
      ::munmap(ptr, size_in_bytes);
      vaddr += size_in_bytes;
      continue; /* slide and retry */
    }

    /* send proposal */
    if (option_DEBUG)
      PLOG("sending vaddr proposal: %p - %ld bytes", ptr, size_in_bytes);

    addr_size_pair offer = {vaddr, size_in_bytes};
    if (write(fd_s2c, &offer, sizeof(addr_size_pair)) != sizeof(addr_size_pair))
      throw General_exception("write failed in uipc_accept_shared_memory (offer)");

    if (option_DEBUG) PLOG("%s", "waiting for response..");

    wait_for_read(fd_c2s, 1);

    char response = 0;
    if (read(fd_c2s, &response, 1) != 1)
      throw General_exception("read failed in uipc_accept_shared_memory (response)");
    assert(response);

    if (response == 'Y') { /* 'Y' signals agreement */
      break;
    }

    /* remove previous mapping and slide along */
    ::munmap(ptr, size_in_bytes);
    vaddr += size_in_bytes;
  } while (1);

  ::close(fd_c2s);
  ::close(fd_s2c);

  if(option_DEBUG)
    PLOG("master: negotiated %p", ptr);
  return ptr;
}

auto Shared_memory::negotiate_addr_connect(const std::string &name) -> mapped_pages
{
  umask(0);

  std::string name_s2c = name + ".s2c";
  std::string name_c2s = name + ".c2s";

  // Wait till master creates the fifo.
  int fd_s2c = -1, fd_c2s = -1;
  while(fd_s2c <= 0){
    fd_s2c = ::open(name_s2c.c_str(), O_RDONLY);
    usleep(100);
  }
  while(fd_c2s <= 0){
    fd_c2s = ::open(name_c2s.c_str(), O_WRONLY);
    usleep(100);
  }
  assert(errno != ENXIO);

  _fifo_names.push_back(name_c2s);
  _fifo_names.push_back(name_s2c);

  void* ptr = nullptr;
  addr_size_pair offer = {0, 0};

  do {
    wait_for_read(fd_s2c,
                  sizeof(addr_size_pair));  // sizeof(void*) + sizeof(size_t));

    if (read(fd_s2c, &offer, sizeof(addr_size_pair)) != sizeof(addr_size_pair))
      throw General_exception(
          "fread failed (offer) in uipc_connect_shared_memory");
      /* Note: throw leaves fs_s2d and fs_c2s open */

    if(option_DEBUG)
      PLOG("got offer %lx - %ld bytes", offer.addr, offer.size);

    assert(offer.size > 0);

    ptr = ::mmap(offer.ptr(),
               offer.size,
               PROT_NONE,
               MAP_SHARED | MAP_ANON,
               0, 0);

    char answer;
    if (ptr != offer.ptr()) {
      ::munmap(ptr, offer.size);
      answer = 'N';
      if (write(fd_c2s, &answer, sizeof(answer)) != sizeof(answer))
        throw General_exception("write failed");
        /* Note: throw leaves fs_s2d and fs_c2s open */
    }
    else {
      answer = 'Y';
      if (write(fd_c2s, &answer, sizeof(answer)) != sizeof(answer))
        throw General_exception("write failed");
        /* Note: throw leaves fs_s2d and fs_c2s open */
      break;
    }
  } while (1);

  ::close(fd_s2c);
  ::close(fd_c2s);

  if(option_DEBUG)
    PLOG("slave: negotiated %p", ptr);
  return {ptr, offer.size / PAGE_SIZE};
}

}  // namespace UIPC
}  // namespace Core
