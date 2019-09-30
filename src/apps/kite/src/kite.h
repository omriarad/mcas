#ifndef __KITE_FILE_HEADERS_H__
#define __KITE_FILE_HEADERS_H__

#define KSB_FILE_MAGIC 0x1B10

#include <stdint.h>

struct File_binary_header_t {
  uint32_t magic;
  uint16_t K;
  bool collecting_origins;
  uint8_t resvd;
  uint64_t genome_count;
  uint64_t single_genome_id; /* if its a single genome, we hold the id here so
                                we don't need per-kmer origin vector */
  uint64_t kmer_count;
  uint64_t kmer_data_offset; /* offset to compressed data, 0 indicates no
                                compression */
} __attribute__((packed));

struct File_binary_kmer_t {
  uint32_t genome_vector_len;
  uint32_t close_to_vector_len;
  uint64_t n_genomes;
  uint64_t counter;
  unsigned char data[0]; /* kmer in 4bpb form | genome vector 64b uints |
                   close_to_vector 64b uints */
} __attribute__((packed));

#endif