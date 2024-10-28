// Copyright (c) 2009-2011, Tor M. Aamodt, Wilson W.L. Fung
// The University of British Columbia
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice, this
// list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice,
// this list of conditions and the following disclaimer in the documentation
// and/or other materials provided with the distribution. Neither the name of
// The University of British Columbia nor the names of its contributors may be
// used to endorse or promote products derived from this software without
// specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#ifndef memory_h_INCLUDED
#define memory_h_INCLUDED

#include "../abstract_hardware_model.h"

#include "../tr1_hash_map.h"
/**
 * "../tr1_hash_map.h"中有如下定义：
 * #define tr1_hash_map std::unordered_map   std::unordered_map 重命名为
 * tr1_hash_map #define tr1_hash_map_ismap 0              设置
 * tr1_hash_map_ismap = 0
 */
#define mem_map tr1_hash_map
#if tr1_hash_map_ismap == 1
#define MEM_MAP_RESIZE(hash_size)
#else
#define MEM_MAP_RESIZE(hash_size) (m_data.rehash(hash_size))
#endif

#include <assert.h>
#include <map>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>

typedef address_type mem_addr_t;

/*定义了内存块的大小，为 4 KB */
#define MEM_BLOCK_SIZE (4 * 1024)

/**
 * 模板类，实现一个 BSIZE 大小的可读写的内存字节数组
 */
template <unsigned BSIZE> class mem_storage {
public:
    /*拷贝构造 */
    mem_storage(const mem_storage &another) {
        m_data = (unsigned char *)calloc(1, BSIZE);
        memcpy(m_data, another.m_data, BSIZE);
    }
    /*默认构造 */
    mem_storage() { m_data = (unsigned char *)calloc(1, BSIZE); }
    /*析构函数 */
    ~mem_storage() { free(m_data); }

    /*把data位置的length大小数据写进字节数组offset的位置 */
    void write(unsigned offset, size_t length, const unsigned char *data) {
        assert(offset + length <= BSIZE);
        memcpy(m_data + offset, data, length);
    }

    /*把字节数组offset的位置开始的length大小的数组写入data*/
    void read(unsigned offset, size_t length, unsigned char *data) const {
        assert(offset + length <= BSIZE);
        memcpy(data, m_data + offset, length);
    }

    /*打印存储中的内容 */
    void print(const char *format, FILE *fout) const {
        unsigned int *i_data = (unsigned int *)m_data;
        for (int d = 0; d < (BSIZE / sizeof(unsigned int)); d++) {
            if (d % 1 == 0) {
                fprintf(fout, "\n");
            }
            fprintf(fout, format, i_data[d]);
            fprintf(fout, " ");
        }
        fprintf(fout, "\n");
        fflush(fout);
    }

private:
    /*无效变量 */
    unsigned m_nbytes;
    /*m_data是指向该page的第一个字节的指针*/
    unsigned char *m_data;
};

class ptx_thread_info;
class ptx_instruction;

/**
 * 用于function model的memory space的建模，提供虚函数，接口
 */
class memory_space {
public:
    virtual ~memory_space() {}
    virtual void write(mem_addr_t addr, size_t length, const void *data,
                       ptx_thread_info *thd, const ptx_instruction *pI) = 0;
    virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                            const void *data) = 0;
    virtual void read(mem_addr_t addr, size_t length, void *data) const = 0;
    virtual void print(const char *format, FILE *fout) const = 0;
    virtual void set_watch(addr_t addr, unsigned watchpoint) = 0;
};

/**
 * memory space具体实现，采用hash表实现，即给定地址mem_addr_t，hash出对应的连续字节block的 mem_storage
 */
template <unsigned BSIZE> class memory_space_impl : public memory_space {
public:
    memory_space_impl(std::string name, unsigned hash_size);

    virtual void write(mem_addr_t addr, size_t length, const void *data,
                       ptx_thread_info *thd, const ptx_instruction *pI);
    virtual void write_only(mem_addr_t index, mem_addr_t offset, size_t length,
                            const void *data);
    virtual void read(mem_addr_t addr, size_t length, void *data) const;
    virtual void print(const char *format, FILE *fout) const;

    virtual void set_watch(addr_t addr, unsigned watchpoint);

private:
    void read_single_block(mem_addr_t blk_idx, mem_addr_t addr, size_t length,
                           void *data) const;
    std::string m_name;
    /*block的二进制位数，如果block size = 4 byte ，则m_log2_block_size = 2 */
    unsigned m_log2_block_size;
    typedef mem_map<mem_addr_t, mem_storage<BSIZE>> map_t;
    map_t m_data;
    std::map<unsigned, mem_addr_t> m_watchpoints;
};

#endif
