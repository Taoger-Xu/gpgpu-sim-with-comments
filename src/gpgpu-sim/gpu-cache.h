// Copyright (c) 2009-2021, Tor M. Aamodt, Tayler Hetherington, Vijay Kandiah,
// Nikos Hardavellas, Mahmoud Khairy, Junrui Pan, Timothy G. Rogers The
// University of British Columbia, Northwestern University, Purdue University
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//    list of conditions and the following disclaimer;
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution;
// 3. Neither the names of The University of British Columbia, Northwestern
//    University nor the names of their contributors may be used to
//    endorse or promote products derived from this software without specific
//    prior written permission.
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

#ifndef GPU_CACHE_H
#define GPU_CACHE_H

#include "../abstract_hardware_model.h"
#include "../tr1_hash_map.h"
#include "gpu-misc.h"
#include "mem_fetch.h"
#include <stdio.h>
#include <stdlib.h>

#include "addrdec.h"
#include <iostream>

#define MAX_DEFAULT_CACHE_SIZE_MULTIBLIER 4

/**
 * INVALID：说明cache block当前存储的数据无效
 
 * VALID：说明cache block存储的数据有效，可以立即为cache hit提供数据
  
 * RESERVED： 当一个cache block被分配用来填充上次的之前因为miss而去low level memory取回的数据时，如果其他的request也命中该block
 * 但是该cache block 的状态 m_status 被设置为 RESERVED，从而不需要向下级发送memory request
 * 
 * MODIFIED: 该 block 的数据已经被其他线程修改, 如果当前访问也是写操作的话即为命中;如果是读操作,需要通过
 */
enum cache_block_state { INVALID = 0, RESERVED, VALID, MODIFIED };

/**
 * HIT：tag命中，当前可以取出数据
 * HIT_RESERVED：tag命中，表示当前cache的数据块正在回填，不需要向下级memory发送请求，只需要等待data response
 * MISS：tag不命中，需要向下级发送memory request
 * RESERVATION_FAIL：cache miss，并且当前cache不能分配更多的资源比如mshr去处理该miss
 * SECTOR_MISS：tag命中，当前的读请求无法读取当前为MODIFIED的sector的data，需要向下级发送read request取回数据；
 *                      或者当前sector为invalid，但是sector所处的line含有不是invalid的sector
 */
enum cache_request_status {
    HIT = 0,
    HIT_RESERVED,
    MISS,
    RESERVATION_FAIL,
    SECTOR_MISS,
    MSHR_HIT,
    NUM_CACHE_REQUEST_STATUS
};

enum cache_reservation_fail_reason {
    LINE_ALLOC_FAIL = 0, // all line are reserved
    MISS_QUEUE_FULL,     // MISS queue (i.e. interconnect or DRAM) is full
    MSHR_ENRTY_FAIL,
    MSHR_MERGE_ENRTY_FAIL,
    MSHR_RW_PENDING,
    NUM_CACHE_RESERVATION_FAIL_STATUS
};

enum cache_event_type {
    WRITE_BACK_REQUEST_SENT,
    READ_REQUEST_SENT,
    WRITE_REQUEST_SENT,
    WRITE_ALLOCATE_SENT
};

enum cache_gpu_level {
    L1_GPU_CACHE = 0,
    L2_GPU_CACHE,
    OTHER_GPU_CACHE,
    NUM_CACHE_GPU_LEVELS
};

struct evicted_block_info {
    new_addr_type m_block_addr;
    unsigned m_modified_size;
    mem_access_byte_mask_t m_byte_mask;
    mem_access_sector_mask_t m_sector_mask;
    evicted_block_info() {
        m_block_addr = 0;
        m_modified_size = 0;
        m_byte_mask.reset();
        m_sector_mask.reset();
    }
    void set_info(new_addr_type block_addr, unsigned modified_size) {
        m_block_addr = block_addr;
        m_modified_size = modified_size;
    }
    void set_info(new_addr_type block_addr, unsigned modified_size,
                  mem_access_byte_mask_t byte_mask,
                  mem_access_sector_mask_t sector_mask) {
        m_block_addr = block_addr;
        m_modified_size = modified_size;
        m_byte_mask = byte_mask;
        m_sector_mask = sector_mask;
    }
};

struct cache_event {
    enum cache_event_type m_cache_event_type;
    evicted_block_info m_evicted_block; // if it was write_back event, fill the
                                        // the evicted block info

    cache_event(enum cache_event_type m_cache_event) {
        m_cache_event_type = m_cache_event;
    }

    cache_event(enum cache_event_type cache_event,
                evicted_block_info evicted_block) {
        m_cache_event_type = cache_event;
        m_evicted_block = evicted_block;
    }
};

const char *cache_request_status_str(enum cache_request_status status);

/**
 * cache_block_t只建模cache block的tag，不建模data block，所有的成员函数都等待子类重写
 * 初始化设置cache block的tag位为0。
 * Memory  |————————|——————————|————————|
 * Address    Tag       Set    Byte Offset
 */
struct cache_block_t {
    cache_block_t() {
        m_tag = 0;
        m_block_addr = 0;
    }

    /*修改cacheline的状态*/
    virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                          unsigned time,mem_access_sector_mask_t sector_mask) = 0;
    
    /*请求得到响应，回填cache line */
    virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                      mem_access_byte_mask_t byte_mask) = 0;
    
    virtual bool is_invalid_line() = 0;
    virtual bool is_valid_line() = 0;
    virtual bool is_reserved_line() = 0;
    virtual bool is_modified_line() = 0;

    virtual enum cache_block_state
    get_status(mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_status(enum cache_block_state m_status,
                            mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_byte_mask(mem_fetch *mf) = 0;
    virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) = 0;
    virtual mem_access_byte_mask_t get_dirty_byte_mask() = 0;
    virtual mem_access_sector_mask_t get_dirty_sector_mask() = 0;
    virtual unsigned long long get_last_access_time() = 0;
    virtual void set_last_access_time(unsigned long long time,
                                      mem_access_sector_mask_t sector_mask) = 0;
    virtual unsigned long long get_alloc_time() = 0;
    virtual void set_ignore_on_fill(bool m_ignore,
                                    mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_modified_on_fill(bool m_modified,
                                      mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_readable_on_fill(bool readable,
                                      mem_access_sector_mask_t sector_mask) = 0;
    virtual void set_byte_mask_on_fill(bool m_modified) = 0;
    virtual unsigned get_modified_size() = 0;
    virtual void set_m_readable(bool readable,
                                mem_access_sector_mask_t sector_mask) = 0;
    virtual bool is_readable(mem_access_sector_mask_t sector_mask) = 0;
    virtual void print_status() = 0;
    virtual ~cache_block_t() {}

    /*无论是line cache还是sector cache，一个cache line都只有一个tag */
    new_addr_type m_tag;
    new_addr_type m_block_addr;
};

// 1. 如果是 Line Cache：按固定大小的cache block来存储数据
//  4 sets, 6 ways, cache blocks are not split to sectors.
//  |-----------------------------------|
//  |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
//  |-----------------------------------|
//  |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
//  |-----------------------------------|
//  |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
//  |-----------------------------------|
//  |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
//  |--------------|--------------------|
//                 |--------> 20 is the cache block index
struct line_cache_block : public cache_block_t {
    /*初始化cacheLine状态为INVALID，m_readable为true*/
    line_cache_block() {
        m_alloc_time = 0;
        m_fill_time = 0;
        m_last_access_time = 0;
        m_status = INVALID;
        m_ignore_on_fill_status = false;
        m_set_modified_on_fill = false;
        m_set_readable_on_fill = false;
        m_readable = true;
    }

    /*修改当前cache block对应的tag等，状态修改为RESERVED表示等待数据回填*/
    void allocate(new_addr_type tag, new_addr_type block_addr, unsigned time,
                  mem_access_sector_mask_t sector_mask) {
        m_tag = tag;
        m_block_addr = block_addr;
        m_alloc_time = time;
        m_last_access_time = time;
        /*等待数据回填 */
        m_fill_time = 0;
        /*把cacheLine状态修改为RESERVED */
        m_status = RESERVED;
        m_ignore_on_fill_status = false;
        m_set_modified_on_fill = false;
        m_set_readable_on_fill = false;
        m_set_byte_mask_on_fill = false;
    }

    /**
     * 数据回填时修改fill_time
     */
    virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                      mem_access_byte_mask_t byte_mask) {
        // if(!m_ignore_on_fill_status)
        //	assert( m_status == RESERVED );

        m_status = m_set_modified_on_fill ? MODIFIED : VALID;

        if (m_set_readable_on_fill)
            m_readable = true;
        if (m_set_byte_mask_on_fill)
            set_byte_mask(byte_mask);

        m_fill_time = time;
    }

    virtual bool is_invalid_line() { return m_status == INVALID; }
    virtual bool is_valid_line() { return m_status == VALID; }
    virtual bool is_reserved_line() { return m_status == RESERVED; }
    virtual bool is_modified_line() { return m_status == MODIFIED; }

    virtual enum cache_block_state
    get_status(mem_access_sector_mask_t sector_mask) {
        return m_status;
    }
    virtual void set_status(enum cache_block_state status,
                            mem_access_sector_mask_t sector_mask) {
        m_status = status;
    }
    virtual void set_byte_mask(mem_fetch *mf) {
        m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
    }
    virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
        m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
    }
    virtual mem_access_byte_mask_t get_dirty_byte_mask() {
        return m_dirty_byte_mask;
    }
    virtual mem_access_sector_mask_t get_dirty_sector_mask() {
        mem_access_sector_mask_t sector_mask;
        if (m_status == MODIFIED)
            sector_mask.set();
        return sector_mask;
    }
    virtual unsigned long long get_last_access_time() {
        return m_last_access_time;
    }
    virtual void set_last_access_time(unsigned long long time,
                                      mem_access_sector_mask_t sector_mask) {
        m_last_access_time = time;
    }
    virtual unsigned long long get_alloc_time() { return m_alloc_time; }
    virtual void set_ignore_on_fill(bool m_ignore,
                                    mem_access_sector_mask_t sector_mask) {
        m_ignore_on_fill_status = m_ignore;
    }
    virtual void set_modified_on_fill(bool m_modified,
                                      mem_access_sector_mask_t sector_mask) {
        m_set_modified_on_fill = m_modified;
    }
    virtual void set_readable_on_fill(bool readable,
                                      mem_access_sector_mask_t sector_mask) {
        m_set_readable_on_fill = readable;
    }
    virtual void set_byte_mask_on_fill(bool m_modified) {
        m_set_byte_mask_on_fill = m_modified;
    }
    virtual unsigned get_modified_size() {
        return SECTOR_CHUNCK_SIZE * SECTOR_SIZE; // i.e. cache line size
    }
    virtual void set_m_readable(bool readable,
                                mem_access_sector_mask_t sector_mask) {
        m_readable = readable;
    }
    virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
        return m_readable;
    }
    virtual void print_status() {
        printf("m_block_addr is %llu, status = %u\n", m_block_addr, m_status);
    }

private:
    /*分配时间*/
    unsigned long long m_alloc_time;
    /*最后一次访问时间 */
    unsigned long long m_last_access_time;
    /*填充数据的时间 */
    unsigned long long m_fill_time;
    /*cacheLine的状态*/
    cache_block_state m_status;
    bool m_ignore_on_fill_status;

    /*数据回填时的状态改变要求*/
    bool m_set_modified_on_fill;
    bool m_set_readable_on_fill;
    bool m_set_byte_mask_on_fill;

    /*标记该cache block是否可读，为false即表示需要向下级发送read request取回数据 */
    bool m_readable;

    /*支持write的粒度为1 byte的write mask */
    mem_access_byte_mask_t m_dirty_byte_mask;
};

// 2. 如果是 Sector Cache：每个cache block被进一步细分为若干个sector或称为“扇区”
//  4 sets, 6 ways, each cache block is split to 4 sectors.
//  |-----------------------------------|
//  |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
//  |-----------------------------------|
//  |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
//  |-----------------------------------|
//  |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
//  |-----------------------------------|
//  |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
//  |-----------/-----\-----------------|
//             /       \
//            /         \--------> 20 is the cache block index
//           /           \
//          /             \
//         |---------------|
//         | 3 | 2 | 1 | 0 | // a block contains 4 sectors
//         |---------|-----|
//                   |--------> 1 is the sector offset in the 20-th cache block

/**
 * 当发生对一个扇区的未命中时，缓存会将一个现存的扇区逐出，将地址标签（tag）设置为新扇区的地址，并且仅加载一个子扇区。
 * 当发生对一个子扇区的未命中，但包含该子扇区的扇区已在缓存中时，只有所需的那个子扇区会被加载
 */
struct sector_cache_block : public cache_block_t {
    
    sector_cache_block() { init(); }

    /**
     * 初始化所有扇区状态为INVALID，设置m_readable[i] = true
     */
    void init() {
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            m_sector_alloc_time[i] = 0;
            m_sector_fill_time[i] = 0;
            m_last_sector_access_time[i] = 0;
            m_status[i] = INVALID;
            m_ignore_on_fill_status[i] = false;
            m_set_modified_on_fill[i] = false;
            m_set_readable_on_fill[i] = false;
            m_readable[i] = true;
        }
        m_line_alloc_time = 0;
        m_line_last_access_time = 0;
        m_line_fill_time = 0;
        m_dirty_byte_mask.reset();
    }

    virtual void allocate(new_addr_type tag, new_addr_type block_addr,
                          unsigned time, mem_access_sector_mask_t sector_mask) {
        allocate_line(tag, block_addr, time, sector_mask);
    }

    /**
     * 设置对应sector_mask的扇区状态为RESERVED
     */
    void allocate_line(new_addr_type tag, new_addr_type block_addr,
                       unsigned time, mem_access_sector_mask_t sector_mask) {
        // allocate a new line
        // assert(m_block_addr != 0 && m_block_addr != block_addr);
        init();
        m_tag = tag;
        m_block_addr = block_addr;

        /*得到扇区索引*/
        unsigned sidx = get_sector_index(sector_mask);

        // set sector stats
        m_sector_alloc_time[sidx] = time;
        m_last_sector_access_time[sidx] = time;
        m_sector_fill_time[sidx] = 0;
        m_status[sidx] = RESERVED;
        m_ignore_on_fill_status[sidx] = false;
        m_set_modified_on_fill[sidx] = false;
        m_set_readable_on_fill[sidx] = false;
        m_set_byte_mask_on_fill = false;

        // set line stats
        m_line_alloc_time = time; // only set this for the first allocated sector
        m_line_last_access_time = time;
        m_line_fill_time = 0;
    }

    /*把对应sector的状态修改为RESERVED */
    void allocate_sector(unsigned time, mem_access_sector_mask_t sector_mask) {
        // allocate invalid sector of this allocated valid line
        assert(is_valid_line());
        unsigned sidx = get_sector_index(sector_mask);

        // set sector stats
        m_sector_alloc_time[sidx] = time;
        m_last_sector_access_time[sidx] = time;
        m_sector_fill_time[sidx] = 0;
        if (m_status[sidx] == MODIFIED) // this should be the case only for fetch-on-write policy //TO DO
            m_set_modified_on_fill[sidx] = true;
        else
            m_set_modified_on_fill[sidx] = false;

        m_set_readable_on_fill[sidx] = false;

        m_status[sidx] = RESERVED;
        m_ignore_on_fill_status[sidx] = false;
        // m_set_modified_on_fill[sidx] = false;
        m_readable[sidx] = true;

        // set line stats
        m_line_last_access_time = time;
        m_line_fill_time = 0;
    }

    /**
     * 修改对应sector的fill time
     */
    virtual void fill(unsigned time, mem_access_sector_mask_t sector_mask,
                      mem_access_byte_mask_t byte_mask) {
        unsigned sidx = get_sector_index(sector_mask);

        //	if(!m_ignore_on_fill_status[sidx])
        //	         assert( m_status[sidx] == RESERVED );
        m_status[sidx] = m_set_modified_on_fill[sidx] ? MODIFIED : VALID;

        if (m_set_readable_on_fill[sidx]) {
            m_readable[sidx] = true;
            m_set_readable_on_fill[sidx] = false;
        }
        if (m_set_byte_mask_on_fill)
            set_byte_mask(byte_mask);

        m_sector_fill_time[sidx] = time;
        m_line_fill_time = time;
    }
    
    /*所有的sector都为INVALID才返回true */
    virtual bool is_invalid_line() {
        // all the sectors should be invalid
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] != INVALID)
                return false;
        }
        return true;
    }

    /*只要有一个sector不为INVALID即返回true */
    virtual bool is_valid_line() { return !(is_invalid_line()); }

    /*只要有一个sector为RESERVED即返回true*/
    virtual bool is_reserved_line() {
        // if any of the sector is reserved, then the line is reserved
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == RESERVED)
                return true;
        }
        return false;
    }

    /*只要有一个sector为MODIFIED即返回true*/
    virtual bool is_modified_line() {
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == MODIFIED)
                return true;
        }
        return false;
    }

    virtual enum cache_block_state

    /*得到该sector对应的状态 */
    get_status(mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);

        return m_status[sidx];
    }

    virtual void set_status(enum cache_block_state status,
                            mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        m_status[sidx] = status;
    }

    virtual void set_byte_mask(mem_fetch *mf) {
        m_dirty_byte_mask = m_dirty_byte_mask | mf->get_access_byte_mask();
    }
    virtual void set_byte_mask(mem_access_byte_mask_t byte_mask) {
        m_dirty_byte_mask = m_dirty_byte_mask | byte_mask;
    }
    virtual mem_access_byte_mask_t get_dirty_byte_mask() {
        return m_dirty_byte_mask;
    }
    virtual mem_access_sector_mask_t get_dirty_sector_mask() {
        mem_access_sector_mask_t sector_mask;
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; i++) {
            if (m_status[i] == MODIFIED)
                sector_mask.set(i);
        }
        return sector_mask;
    }
    virtual unsigned long long get_last_access_time() {
        return m_line_last_access_time;
    }

    virtual void set_last_access_time(unsigned long long time,
                                      mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);

        m_last_sector_access_time[sidx] = time;
        m_line_last_access_time = time;
    }

    virtual unsigned long long get_alloc_time() { return m_line_alloc_time; }

    virtual void set_ignore_on_fill(bool m_ignore,
                                    mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        m_ignore_on_fill_status[sidx] = m_ignore;
    }

    virtual void set_modified_on_fill(bool m_modified,
                                      mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        m_set_modified_on_fill[sidx] = m_modified;
    }
    virtual void set_byte_mask_on_fill(bool m_modified) {
        m_set_byte_mask_on_fill = m_modified;
    }

    virtual void set_readable_on_fill(bool readable,
                                      mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        m_set_readable_on_fill[sidx] = readable;
    }
    
    /** */
    virtual void set_m_readable(bool readable,
                                mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        m_readable[sidx] = readable;
    }

    /* */
    virtual bool is_readable(mem_access_sector_mask_t sector_mask) {
        unsigned sidx = get_sector_index(sector_mask);
        return m_readable[sidx];
    }

    virtual unsigned get_modified_size() {
        unsigned modified = 0;
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (m_status[i] == MODIFIED)
                modified++;
        }
        return modified * SECTOR_SIZE;
    }

    virtual void print_status() {
        printf("m_block_addr is %llu, status = %u %u %u %u\n", m_block_addr,
               m_status[0], m_status[1], m_status[2], m_status[3]);
    }

private:
    unsigned m_sector_alloc_time[SECTOR_CHUNCK_SIZE];
    unsigned m_last_sector_access_time[SECTOR_CHUNCK_SIZE];
    unsigned m_sector_fill_time[SECTOR_CHUNCK_SIZE];

    unsigned m_line_alloc_time;
    unsigned m_line_last_access_time;
    unsigned m_line_fill_time;

    cache_block_state m_status[SECTOR_CHUNCK_SIZE];
    bool m_ignore_on_fill_status[SECTOR_CHUNCK_SIZE];
    bool m_set_modified_on_fill[SECTOR_CHUNCK_SIZE];
    bool m_set_readable_on_fill[SECTOR_CHUNCK_SIZE];
    bool m_set_byte_mask_on_fill;

    /*表示当前sector不可读，需要向下级发送一个read request取回对应sector的数据 */
    bool m_readable[SECTOR_CHUNCK_SIZE];

    /*针对写粒度为1 byte的 write mask */
    mem_access_byte_mask_t m_dirty_byte_mask;

    /*  给出sector_mask，给出对应的扇区索引
        例如0001返回0，0010返回1，0100返回2，1000返回3
        每一个tag只能命中一个sector
     */
    unsigned get_sector_index(mem_access_sector_mask_t sector_mask) {
        assert(sector_mask.count() == 1);
        for (unsigned i = 0; i < SECTOR_CHUNCK_SIZE; ++i) {
            if (sector_mask.to_ulong() & (1 << i))
                return i;
        }
        return SECTOR_CHUNCK_SIZE; // error
    }
};


enum replacement_policy_t { LRU, FIFO };

enum write_policy_t {
    READ_ONLY,
    WRITE_BACK,
    WRITE_THROUGH,
    WRITE_EVICT,
    LOCAL_WB_GLOBAL_WT
};

/**
  allocate-on-miss（简写为on-miss）和allocate-on-fill（简写为on-fill）是两种cache resources分配策略
    * allocate-on-miss分配的资源为:
    *  - a cache line slot, a MSHR, and a miss queue entry
    * 即在访问miss的就需要选择一个cache line作为reserved line一直等待data回填
    * allocate-on-fill分配的资源为:
    *  - a MSHR and a miss queue entry
    * 即allocate-on-fill只有required data返回时才选择 victim cache line slot进行驱逐  
*/
enum allocation_policy_t { ON_MISS, ON_FILL, STREAMING };

/**
 * 对于当前cache的write miss是否分配cacheline将策略分为write-allocate和no-write-allocate
 * 对于当前的cache的write miss是否从下级内存取回将要写入的block将策略分为：fetch-on-write和no-fetch-on-write
 * fetch-on-write的策略必须等待下级内存取回block后才能处理写入
 * 
 * 对于L2的sector cache：
 *  In fetch-on-write, when we
    write to a single byte of a sector, the L2 fetches the whole
    sector then merges the written portion to the sector and sets the
    sector as modified. In the write-validate policy, no read fetch is
    required, instead each sector has a bit-wise write-mask. When
    a write to a single byte is received, it writes the byte to the
    sector, sets the corresponding write bit and sets the sector as
    valid and modified.
    When a modified cache line is evicted, the
    cache line is written back to the memory along with the write mask
 */
enum write_allocate_policy_t {
    NO_WRITE_ALLOCATE,
    WRITE_ALLOCATE,
    FETCH_ON_WRITE,
    LAZY_FETCH_ON_READ
};

/*在GV100的MSHR type上，L1D为ASSOC，L2D为ASSOC，L1I为SECTOR_ASSOC*/
enum mshr_config_t {
    TEX_FIFO,        // Tex cache
    ASSOC,           // normal cache
    SECTOR_TEX_FIFO, // Tex cache sends requests to high-level sector cache
    SECTOR_ASSOC     // normal cache sends requests to high-level sector cache
};

enum set_index_function {
    LINEAR_SET_FUNCTION = 0,
    BITWISE_XORING_FUNCTION,
    HASH_IPOLY_FUNCTION,
    FERMI_HASH_SET_FUNCTION,
    CUSTOM_SET_FUNCTION
};

enum cache_type { NORMAL = 0, SECTOR };

#define MAX_WARP_PER_SHADER 64
#define INCT_TOTAL_BUFFER 64
#define L2_TOTAL 64
#define MAX_WARP_PER_SHADER 64
#define MAX_WARP_PER_SHADER 64

class cache_config {
public:
    cache_config() {
        m_valid = false;
        m_disabled = false;
        m_config_string = NULL; // set by option parser
        m_config_stringPrefL1 = NULL;
        m_config_stringPrefShared = NULL;
        m_data_port_width = 0;
        m_set_index_function = LINEAR_SET_FUNCTION;
        m_is_streaming = false;
        m_wr_percent = 0;
    }
    void init(char *config, FuncCache status) {
        cache_status = status;
        assert(config);
        char ct, rp, wp, ap, mshr_type, wap, sif;

        int ntok = sscanf(
            config, "%c:%u:%u:%u,%c:%c:%c:%c:%c,%c:%u:%u,%u:%u,%u", &ct,
            &m_nset, &m_line_sz, &m_assoc, &rp, &wp, &ap, &wap, &sif,
            &mshr_type, &m_mshr_entries, &m_mshr_max_merge, &m_miss_queue_size,
            &m_result_fifo_entries, &m_data_port_width);

        if (ntok < 12) {
            if (!strcmp(config, "none")) {
                m_disabled = true;
                return;
            }
            exit_parse_error();
        }

        switch (ct) {
        case 'N':
            m_cache_type = NORMAL;
            break;
        case 'S':
            m_cache_type = SECTOR;
            break;
        default:
            exit_parse_error();
        }
        switch (rp) {
        case 'L':
            m_replacement_policy = LRU;
            break;
        case 'F':
            m_replacement_policy = FIFO;
            break;
        default:
            exit_parse_error();
        }
        switch (wp) {
        case 'R':
            m_write_policy = READ_ONLY;
            break;
        case 'B':
            m_write_policy = WRITE_BACK;
            break;
        case 'T':
            m_write_policy = WRITE_THROUGH;
            break;
        case 'E':
            m_write_policy = WRITE_EVICT;
            break;
        case 'L':
            m_write_policy = LOCAL_WB_GLOBAL_WT;
            break;
        default:
            exit_parse_error();
        }
        switch (ap) {
        case 'm':
            m_alloc_policy = ON_MISS;
            break;
        case 'f':
            m_alloc_policy = ON_FILL;
            break;
        case 's':
            m_alloc_policy = STREAMING;
            break;
        default:
            exit_parse_error();
        }
        if (m_alloc_policy == STREAMING) {
            /*
            For streaming cache:
            (1) we set the alloc policy to be on-fill to remove all
            line_alloc_fail stalls. if the whole memory is allocated to the L1
            cache, then make the allocation to be on_MISS otherwise, make it
            ON_FILL to eliminate line allocation fails. i.e. MSHR throughput is
            the same, independent on the L1 cache size/associativity So, we set
            the allocation policy per kernel basis, see shader.cc, max_cta()
            function

            (2) We also set the MSHRs to be equal to max
            allocated cache lines. This is possible by moving TAG to be shared
            between cache line and MSHR enrty (i.e. for each cache line, there
            is an MSHR rntey associated with it). This is the easiest think we
            can think of to model (mimic) L1 streaming cache in Pascal and Volta

            For more information about streaming cache, see:
            http://on-demand.gputechconf.com/gtc/2017/presentation/s7798-luke-durant-inside-volta.pdf
            https://ieeexplore.ieee.org/document/8344474/
            */
            m_is_streaming = true;
            m_alloc_policy = ON_FILL;
        }
        switch (mshr_type) {
        case 'F':
            m_mshr_type = TEX_FIFO;
            assert(ntok == 14);
            break;
        case 'T':
            m_mshr_type = SECTOR_TEX_FIFO;
            assert(ntok == 14);
            break;
        case 'A':
            m_mshr_type = ASSOC;
            break;
        case 'S':
            m_mshr_type = SECTOR_ASSOC;
            break;
        default:
            exit_parse_error();
        }
        m_line_sz_log2 = LOGB2(m_line_sz);
        m_nset_log2 = LOGB2(m_nset);
        m_valid = true;
        m_atom_sz = (m_cache_type == SECTOR) ? SECTOR_SIZE : m_line_sz;
        m_sector_sz_log2 = LOGB2(SECTOR_SIZE);
        original_m_assoc = m_assoc;

        // For more details about difference between FETCH_ON_WRITE and WRITE
        // VALIDAE policies Read: Jouppi, Norman P. "Cache write policies and
        // performance". ISCA 93. WRITE_ALLOCATE is the old write policy in
        // GPGPU-sim 3.x, that send WRITE and READ for every write request
        switch (wap) {
        case 'N':
            m_write_alloc_policy = NO_WRITE_ALLOCATE;
            break;
        case 'W':
            m_write_alloc_policy = WRITE_ALLOCATE;
            break;
        case 'F':
            m_write_alloc_policy = FETCH_ON_WRITE;
            break;
        case 'L':
            m_write_alloc_policy = LAZY_FETCH_ON_READ;
            break;
        default:
            exit_parse_error();
        }

        // detect invalid configuration
        if ((m_alloc_policy == ON_FILL || m_alloc_policy == STREAMING) and
            m_write_policy == WRITE_BACK) {
            // A writeback cache with allocate-on-fill policy will inevitably
            // lead to deadlock: The deadlock happens when an incoming
            // cache-fill evicts a dirty line, generating a writeback request.
            // If the memory subsystem is congested, the interconnection network
            // may not have sufficient buffer for the writeback request.  This
            // stalls the incoming cache-fill.  The stall may propagate through
            // the memory subsystem back to the output port of the same core,
            // creating a deadlock where the wrtieback request and the incoming
            // cache-fill are stalling each other.
            assert(0 && "Invalid cache configuration: Writeback cache cannot "
                        "allocate new "
                        "line on fill. ");
        }

        if ((m_write_alloc_policy == FETCH_ON_WRITE ||
             m_write_alloc_policy == LAZY_FETCH_ON_READ) &&
            m_alloc_policy == ON_FILL) {
            assert(0 && "Invalid cache configuration: FETCH_ON_WRITE and "
                        "LAZY_FETCH_ON_READ "
                        "cannot work properly with ON_FILL policy. Cache must "
                        "be ON_MISS. ");
        }

        if (m_cache_type == SECTOR) {
            bool cond = m_line_sz / SECTOR_SIZE == SECTOR_CHUNCK_SIZE &&
                        m_line_sz % SECTOR_SIZE == 0;
            if (!cond) {
                std::cerr
                    << "error: For sector cache, the simulator uses hard-coded "
                       "SECTOR_SIZE and SECTOR_CHUNCK_SIZE. The line size "
                       "must be product of both values.\n";
                assert(0);
            }
        }

        // default: port to data array width and granularity = line size
        if (m_data_port_width == 0) {
            m_data_port_width = m_line_sz;
        }
        assert(m_line_sz % m_data_port_width == 0);

        switch (sif) {
        case 'H':
            m_set_index_function = FERMI_HASH_SET_FUNCTION;
            break;
        case 'P':
            m_set_index_function = HASH_IPOLY_FUNCTION;
            break;
        case 'C':
            m_set_index_function = CUSTOM_SET_FUNCTION;
            break;
        case 'L':
            m_set_index_function = LINEAR_SET_FUNCTION;
            break;
        case 'X':
            m_set_index_function = BITWISE_XORING_FUNCTION;
            break;
        default:
            exit_parse_error();
        }
    }
    bool disabled() const { return m_disabled; }
    unsigned get_line_sz() const {
        assert(m_valid);
        return m_line_sz;
    }
    unsigned get_atom_sz() const {
        assert(m_valid);
        return m_atom_sz;
    }
    unsigned get_num_lines() const {
        assert(m_valid);
        return m_nset * m_assoc;
    }
    unsigned get_max_num_lines() const {
        assert(m_valid);
        return get_max_cache_multiplier() * m_nset * original_m_assoc;
    }
    unsigned get_max_assoc() const {
        assert(m_valid);
        return get_max_cache_multiplier() * original_m_assoc;
    }
    void print(FILE *fp) const {
        fprintf(fp, "Size = %d B (%d Set x %d-way x %d byte line)\n",
                m_line_sz * m_nset * m_assoc, m_nset, m_assoc, m_line_sz);
    }

    virtual unsigned set_index(new_addr_type addr) const;

    virtual unsigned get_max_cache_multiplier() const {
        return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
    }

    unsigned hash_function(new_addr_type addr, unsigned m_nset,
                           unsigned m_line_sz_log2, unsigned m_nset_log2,
                           unsigned m_index_function) const;

    new_addr_type tag(new_addr_type addr) const {
        // For generality, the tag includes both index and tag. This allows for
        // more complex set index calculations that can result in different
        // indexes mapping to the same set, thus the full tag + index is
        // required to check for hit/miss. Tag is now identical to the block
        // address.

        // return addr >> (m_line_sz_log2+m_nset_log2);
        return addr & ~(new_addr_type)(m_line_sz - 1);
    }

    /* block_addr为地址addr的tag位+set index位。即除offset位以外的所有位*/
    new_addr_type block_addr(new_addr_type addr) const {
        return addr & ~(new_addr_type)(m_line_sz - 1);
    }

    /* mshr_addr为地址addr的tag位+set index位+sector offset位 */
    new_addr_type mshr_addr(new_addr_type addr) const {
        return addr & ~(new_addr_type)(m_atom_sz - 1);
    }
    enum mshr_config_t get_mshr_type() const { return m_mshr_type; }
    void set_assoc(unsigned n) {
        // set new assoc. L1 cache dynamically resized in Volta
        m_assoc = n;
    }
    unsigned get_nset() const {
        assert(m_valid);
        return m_nset;
    }
    unsigned get_total_size_inKB() const {
        assert(m_valid);
        return (m_assoc * m_nset * m_line_sz) / 1024;
    }
    bool is_streaming() { return m_is_streaming; }
    FuncCache get_cache_status() { return cache_status; }
    void set_allocation_policy(enum allocation_policy_t alloc) {
        m_alloc_policy = alloc;
    }
    char *m_config_string;
    char *m_config_stringPrefL1;
    char *m_config_stringPrefShared;
    FuncCache cache_status;
    unsigned m_wr_percent;
    write_allocate_policy_t get_write_allocate_policy() {
        return m_write_alloc_policy;
    }
    write_policy_t get_write_policy() { return m_write_policy; }

protected:
    void exit_parse_error() {
        printf("GPGPU-Sim uArch: cache configuration parsing error (%s)\n",
               m_config_string);
        abort();
    }

    bool m_valid;
    bool m_disabled;
    unsigned m_line_sz;
    unsigned m_line_sz_log2;
    unsigned m_nset;
    unsigned m_nset_log2;
    unsigned m_assoc;
    unsigned m_atom_sz;
    unsigned m_sector_sz_log2;
    unsigned original_m_assoc;
    bool m_is_streaming;

    enum replacement_policy_t m_replacement_policy; // 'L' = LRU, 'F' = FIFO
    enum write_policy_t m_write_policy; // 'T' = write through, 'B' = write
                                        // back, 'R' = read only
    enum allocation_policy_t
        m_alloc_policy; // 'm' = allocate on miss, 'f' = allocate on fill
    enum mshr_config_t m_mshr_type;
    enum cache_type m_cache_type;

    write_allocate_policy_t
        m_write_alloc_policy; // 'W' = Write allocate, 'N' = No write allocate

    union {
        unsigned m_mshr_entries;
        unsigned m_fragment_fifo_entries;
    };
    union {
        unsigned m_mshr_max_merge;
        unsigned m_request_fifo_entries;
    };
    union {
        unsigned m_miss_queue_size;
        unsigned m_rob_entries;
    };
    unsigned m_result_fifo_entries;
    unsigned
        m_data_port_width; //< number of byte the cache can access per cycle
    enum set_index_function
        m_set_index_function; // Hash, linear, or custom set index function

    friend class tag_array;
    friend class baseline_cache;
    friend class read_only_cache;
    friend class tex_cache;
    friend class data_cache;
    friend class l1_cache;
    friend class l2_cache;
    friend class memory_sub_partition;
};

class l1d_cache_config : public cache_config {
public:
    l1d_cache_config() : cache_config() {}
    unsigned set_bank(new_addr_type addr) const;
    void init(char *config, FuncCache status) {
        l1_banks_byte_interleaving_log2 = LOGB2(l1_banks_byte_interleaving);
        l1_banks_log2 = LOGB2(l1_banks);
        cache_config::init(config, status);
    }
    unsigned l1_latency;
    unsigned l1_banks;
    unsigned l1_banks_log2;
    unsigned l1_banks_byte_interleaving;
    unsigned l1_banks_byte_interleaving_log2;
    unsigned l1_banks_hashing_function;
    unsigned m_unified_cache_size;
    virtual unsigned get_max_cache_multiplier() const {
        // set * assoc * cacheline size. Then convert Byte to KB
        // gpgpu_unified_cache_size is in KB while original_sz is in B
        if (m_unified_cache_size > 0) {
            unsigned original_size =
                m_nset * original_m_assoc * m_line_sz / 1024;
            assert(m_unified_cache_size % original_size == 0);
            return m_unified_cache_size / original_size;
        } else {
            return MAX_DEFAULT_CACHE_SIZE_MULTIBLIER;
        }
    }
};

class l2_cache_config : public cache_config {
public:
    l2_cache_config() : cache_config() {}
    void init(linear_to_raw_address_translation *address_mapping);
    virtual unsigned set_index(new_addr_type addr) const;

private:
    linear_to_raw_address_translation *m_address_mapping;
};

/**
 * tag_array主要用来建模二维的CacheLine，其核心成员为cache_block_t **m_lines
*/
class tag_array {
public:
    /*构造函数 */
    tag_array(cache_config &config, int core_id, int type_id);
    ~tag_array();

    /*处理request相关 */
    enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                    mem_fetch *mf, bool is_write,
                                    bool probe_mode = false) const;
    enum cache_request_status probe(new_addr_type addr, unsigned &idx,
                                    mem_access_sector_mask_t mask,
                                    bool is_write, bool probe_mode = false,
                                    mem_fetch *mf = NULL) const;
    enum cache_request_status access(new_addr_type addr, unsigned time,
                                     unsigned &idx, mem_fetch *mf);
    enum cache_request_status access(new_addr_type addr, unsigned time,
                                     unsigned &idx, bool &wb,
                                     evicted_block_info &evicted,
                                     mem_fetch *mf);

    void fill(new_addr_type addr, unsigned time, mem_fetch *mf, bool is_write);
    void fill(unsigned idx, unsigned time, mem_fetch *mf);
    void fill(new_addr_type addr, unsigned time, mem_access_sector_mask_t mask,
              mem_access_byte_mask_t byte_mask, bool is_write);

    unsigned size() const { return m_config.get_num_lines(); }
    cache_block_t *get_block(unsigned idx) { return m_lines[idx]; }

    void flush();      // flush all written entries
    void invalidate(); // invalidate all entries
    void new_window();

    void print(FILE *stream, unsigned &total_access,
               unsigned &total_misses) const;
    float windowed_miss_rate() const;
    void get_stats(unsigned &total_access, unsigned &total_misses,
                   unsigned &total_hit_res, unsigned &total_res_fail) const;

    void update_cache_parameters(cache_config &config);
    void add_pending_line(mem_fetch *mf);
    void remove_pending_line(mem_fetch *mf);

    void inc_dirty() { m_dirty++; }

protected:
    // This constructor is intended for use only from derived classes that wish
    // to avoid unnecessary memory allocation that takes place in the other
    // tag_array constructor
    tag_array(cache_config &config, int core_id, int type_id,
              cache_block_t **new_lines);
    void init(int core_id, int type_id);

protected:
    /*cache配置 */
    cache_config &m_config;

    /** 
     * For example, 4 sets, 6 ways:
    |  0  |  1  |  2  |  3  |  4  |  5  |  // set_index 0
    |  6  |  7  |  8  |  9  |  10 |  11 |  // set_index 1
    |  12 |  13 |  14 |  15 |  16 |  17 |  // set_index 2
    |  18 |  19 |  20 |  21 |  22 |  23 |  // set_index 3
    |--------> index => cache_block_t *line
        m_lines[index] = &m_lines[set_index * m_config.m_assoc + way_index]
    */
    cache_block_t **m_lines; 

    /*下面是统计数据，使用access()访问时更新*/

    /*访问该cache总的次数 */
    unsigned m_access;
    
    /*访问该cache的miss的次数 */
    unsigned m_miss;

    /*访问cache时tag匹配但是data需要等待下级memory回填的request数量
    number of cache miss that hit a line that is allocated but not filled*/
    unsigned m_pending_hit; 
                            
    
    /*RESERVATION_FAIL失败的次数，即cache无法处理该请求的次数*/
    unsigned m_res_fail;
    unsigned m_sector_miss;

    /*状态为dirty的cache block的数量 */
    unsigned m_dirty;

    // performance counters for calculating the amount of misses within a time window
    unsigned m_prev_snapshot_access;
    unsigned m_prev_snapshot_miss;
    unsigned m_prev_snapshot_pending_hit;

    /*该cache所属的simt core的id */
    int m_core_id; 

    /*该cache的类型：(normal, texture, constant)*/
    int m_type_id; 

    /*标记该cache之前是否被access过，调用access()函数后设为true*/
    bool is_used; 

    /*已弃用*/
    typedef tr1_hash_map<new_addr_type, unsigned> line_table;
    line_table pending_lines;
};

/**
 * 一个memory request如果在cache上miss则会被添加到MSHR上
 * A memory request that misses in the cache is added to the MSHR table 
 * and a fill request is generated if there is no pending request for that cache line
 * mshr_table建模一个固定数量的mshr entry的table
 * 给定一个memory request的addr，通过检查后通过add()将其存到mshr
 * 对于lower memory返回得到的数据，通过mark_ready()和next_access()释放对应的mshr entry
 * 
 * 一个mshr entry用来处理finite maximum number of merged requests，
 * 合并的这些请求位于同一个cacheline
 * 其中mshr_table中mshr entry的数量以及每一个entry可以合并的最多memory
 * request是可以配置的
 */
class mshr_table {
public:
    mshr_table(unsigned num_entries, unsigned max_merged)
        : m_num_entries(num_entries), m_max_merged(max_merged)
#if (tr1_hash_map_ismap == 0)
          ,
          m_data(2 * num_entries)
#endif
    {
    }

    /*检查给定的地址block_addr对应的内存请求是否已经被合并到mshr entry，即有已经发送给lower memory的pending request*/ 
    bool probe(new_addr_type block_addr) const;

    /*检查mshr是否还有空间去处理新的memory request*/
    bool full(new_addr_type block_addr) const;

    /*默认MSHR可以处理当前请求，直接向对应的mshr插入请求，要么分配新的entry，要么插入已有的mshr entry*/
    void add(new_addr_type block_addr, mem_fetch *mf);

    /*如果不能接收新的memory request，这里一直返回false*/
    bool busy() const { return false; }

    /*接收一个新的来自lower level的cache fill response，把cache fill response的信息放入m_current_response */
    void mark_ready(new_addr_type block_addr, bool &has_atomic);

    /*判断之前cache miss 需要的数据已经取回来，即存在新的cache fill response */
    bool access_ready() const { return !m_current_response.empty(); }

    /*取出一个已经ready的memory request，并且弹出m_current_response和释放对应的mshr entry */
    mem_fetch *next_access();

    /*打印信息 */
    void display(FILE *fp) const;
    // Returns true if there is a pending read after write
    bool is_read_after_write_pending(new_addr_type block_addr);

    void check_mshr_parameters(unsigned num_entries, unsigned max_merged) {
        assert(m_num_entries == num_entries &&
               "Change of MSHR parameters between kernels is not allowed");
        assert(m_max_merged == max_merged &&
               "Change of MSHR parameters between kernels is not allowed");
    }

private:
    /**mshr 拥有的mshr entry数量 */
    const unsigned m_num_entries;

    /*一个mshr entry能够合并的 memory requests的数量 */
    const unsigned m_max_merged;

    /*
        mshr entry拥有一系列访存miss的memory requests(mem_fetch*)组成的list，这些请求在等待lower-level
        memory对于同一个block的响应
    */
    struct mshr_entry {
        /* 维护所有合并的memory request */
        std::list<mem_fetch *> m_list;
        /* 合并的memory request是否有atomic操作 */
        bool m_has_atomic;
        mshr_entry() : m_has_atomic(false) {}
    };
    /*tr1_hash_map即是std::unordered_map，将内存块地址（new_addr_type）映射到对应的mshr_entry */
    typedef tr1_hash_map<new_addr_type, mshr_entry> table;
    typedef tr1_hash_map<new_addr_type, mshr_entry> line_table;
    /*mshr的核心数据结构，将内存块地址（new_addr_type）映射到对应的 mshr_entry，建模整个mshr */
    table m_data;
    line_table pending_lines;

    /*处理merged requests花费的周期数*/
    bool m_current_response_ready;

    /*存放已经取回数据的cache block的对应的memory request */
    std::list<new_addr_type> m_current_response;
};

/***************************************************************** Caches
 * *****************************************************************/
///
/// Simple struct to maintain cache accesses, misses, pending hits, and
/// reservation fails.
///
struct cache_sub_stats {
    unsigned long long accesses;
    unsigned long long misses;
    unsigned long long pending_hits;
    unsigned long long res_fails;

    unsigned long long port_available_cycles;
    unsigned long long data_port_busy_cycles;
    unsigned long long fill_port_busy_cycles;

    cache_sub_stats() { clear(); }
    void clear() {
        accesses = 0;
        misses = 0;
        pending_hits = 0;
        res_fails = 0;
        port_available_cycles = 0;
        data_port_busy_cycles = 0;
        fill_port_busy_cycles = 0;
    }
    cache_sub_stats &operator+=(const cache_sub_stats &css) {
        ///
        /// Overloading += operator to easily accumulate stats
        ///
        accesses += css.accesses;
        misses += css.misses;
        pending_hits += css.pending_hits;
        res_fails += css.res_fails;
        port_available_cycles += css.port_available_cycles;
        data_port_busy_cycles += css.data_port_busy_cycles;
        fill_port_busy_cycles += css.fill_port_busy_cycles;
        return *this;
    }

    cache_sub_stats operator+(const cache_sub_stats &cs) {
        ///
        /// Overloading + operator to easily accumulate stats
        ///
        cache_sub_stats ret;
        ret.accesses = accesses + cs.accesses;
        ret.misses = misses + cs.misses;
        ret.pending_hits = pending_hits + cs.pending_hits;
        ret.res_fails = res_fails + cs.res_fails;
        ret.port_available_cycles =
            port_available_cycles + cs.port_available_cycles;
        ret.data_port_busy_cycles =
            data_port_busy_cycles + cs.data_port_busy_cycles;
        ret.fill_port_busy_cycles =
            fill_port_busy_cycles + cs.fill_port_busy_cycles;
        return ret;
    }

    void print_port_stats(FILE *fout, const char *cache_name) const;
};

// Used for collecting AerialVision per-window statistics
struct cache_sub_stats_pw {
    unsigned accesses;
    unsigned write_misses;
    unsigned write_hits;
    unsigned write_pending_hits;
    unsigned write_res_fails;

    unsigned read_misses;
    unsigned read_hits;
    unsigned read_pending_hits;
    unsigned read_res_fails;

    cache_sub_stats_pw() { clear(); }
    void clear() {
        accesses = 0;
        write_misses = 0;
        write_hits = 0;
        write_pending_hits = 0;
        write_res_fails = 0;
        read_misses = 0;
        read_hits = 0;
        read_pending_hits = 0;
        read_res_fails = 0;
    }
    cache_sub_stats_pw &operator+=(const cache_sub_stats_pw &css) {
        ///
        /// Overloading += operator to easily accumulate stats
        ///
        accesses += css.accesses;
        write_misses += css.write_misses;
        read_misses += css.read_misses;
        write_pending_hits += css.write_pending_hits;
        read_pending_hits += css.read_pending_hits;
        write_res_fails += css.write_res_fails;
        read_res_fails += css.read_res_fails;
        return *this;
    }

    cache_sub_stats_pw operator+(const cache_sub_stats_pw &cs) {
        ///
        /// Overloading + operator to easily accumulate stats
        ///
        cache_sub_stats_pw ret;
        ret.accesses = accesses + cs.accesses;
        ret.write_misses = write_misses + cs.write_misses;
        ret.read_misses = read_misses + cs.read_misses;
        ret.write_pending_hits = write_pending_hits + cs.write_pending_hits;
        ret.read_pending_hits = read_pending_hits + cs.read_pending_hits;
        ret.write_res_fails = write_res_fails + cs.write_res_fails;
        ret.read_res_fails = read_res_fails + cs.read_res_fails;
        return ret;
    }
};

///
/// Cache_stats
/// Used to record statistics for each cache.
/// Maintains a record of every 'mem_access_type' and its resulting
/// 'cache_request_status' : [mem_access_type][cache_request_status]
///
class cache_stats {
public:
    cache_stats();
    void clear();
    // Clear AerialVision cache stats after each window
    void clear_pw();
    void inc_stats(int access_type, int access_outcome,
                   unsigned long long streamID);
    // Increment AerialVision cache stats
    void inc_stats_pw(int access_type, int access_outcome,
                      unsigned long long streamID);
    void inc_fail_stats(int access_type, int fail_outcome,
                        unsigned long long streamID);
    enum cache_request_status
    select_stats_status(enum cache_request_status probe,
                        enum cache_request_status access) const;
    unsigned long long &operator()(int access_type, int access_outcome,
                                   bool fail_outcome,
                                   unsigned long long streamID);
    unsigned long long operator()(int access_type, int access_outcome,
                                  bool fail_outcome,
                                  unsigned long long streamID) const;
    cache_stats operator+(const cache_stats &cs);
    cache_stats &operator+=(const cache_stats &cs);
    void print_stats(FILE *fout, unsigned long long streamID,
                     const char *cache_name = "Cache_stats") const;
    void print_fail_stats(FILE *fout, unsigned long long streamID,
                          const char *cache_name = "Cache_fail_stats") const;

    unsigned long long get_stats(enum mem_access_type *access_type,
                                 unsigned num_access_type,
                                 enum cache_request_status *access_status,
                                 unsigned num_access_status) const;
    void get_sub_stats(struct cache_sub_stats &css) const;

    // Get per-window cache stats for AerialVision
    void get_sub_stats_pw(struct cache_sub_stats_pw &css) const;

    void sample_cache_port_utility(bool data_port_busy, bool fill_port_busy);

private:
    bool check_valid(int type, int status) const;
    bool check_fail_valid(int type, int fail) const;

    // CUDA streamID -> cache stats[NUM_MEM_ACCESS_TYPE]
    std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
        m_stats;
    // AerialVision cache stats (per-window)
    std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
        m_stats_pw;
    std::map<unsigned long long, std::vector<std::vector<unsigned long long>>>
        m_fail_stats;

    unsigned long long m_cache_port_available_cycles;
    unsigned long long m_cache_data_port_busy_cycles;
    unsigned long long m_cache_fill_port_busy_cycles;
};

class cache_t {
public:
    virtual ~cache_t() {}
    virtual enum cache_request_status
    access(new_addr_type addr, mem_fetch *mf, unsigned time,
           std::list<cache_event> &events) = 0;

    // accessors for cache bandwidth availability
    virtual bool data_port_free() const = 0;
    virtual bool fill_port_free() const = 0;
};

bool was_write_sent(const std::list<cache_event> &events);
bool was_read_sent(const std::list<cache_event> &events);
bool was_writeallocate_sent(const std::list<cache_event> &events);

/**
 * baseline_cache实现read_only_cache和data_cache通用功能，其子类需要实现自己的access()函数
 */
class baseline_cache : public cache_t {
public:
    baseline_cache(const char *name, cache_config &config, int core_id,
                   int type_id, mem_fetch_interface *memport,
                   enum mem_fetch_status status, enum cache_gpu_level level,
                   gpgpu_sim *gpu)
        : m_config(config),
          m_tag_array(new tag_array(config, core_id, type_id)),
          m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
          m_bandwidth_management(config), m_level(level), m_gpu(gpu) {
        init(name, config, memport, status);
    }

    void init(const char *name, const cache_config &config,
              mem_fetch_interface *memport, enum mem_fetch_status status) {
        m_name = name;
        assert(config.m_mshr_type == ASSOC ||
               config.m_mshr_type == SECTOR_ASSOC);
        m_memport = memport;
        m_miss_queue_status = status;
    }

    virtual ~baseline_cache() { delete m_tag_array; }

    void update_cache_parameters(cache_config &config) {
        m_config = config;
        m_tag_array->update_cache_parameters(config);
        m_mshrs.check_mshr_parameters(config.m_mshr_entries,
                                      config.m_mshr_max_merge);
    }

    /*虚函数，不同的cache实现不同的access函数*/
    virtual enum cache_request_status
    access(new_addr_type addr, mem_fetch *mf, unsigned time,
           std::list<cache_event> &events) = 0;

    /**
     * 运行一个cycle，把本周期能处理cache miss的request发送给下级请求
     * 即把mem_fetch从m_miss_queue移动到m_memport
     * */
    void cycle();

    // Interface for response from lower memory level (model bandwidth restictions in caller)
    /*用于处理来自lower memory level的内存响应 */
    void fill(mem_fetch *mf, unsigned time);

    /// Checks if mf is waiting to be filled by lower memory level
    bool waiting_for_fill(mem_fetch *mf);

    /*之前cache miss的data已经返回*/
    bool access_ready() const { return m_mshrs.access_ready(); }
    
    /*得到之前因为miss而在mshr中保留，但是其data已经取回的memory request的信息*/
    mem_fetch *next_access() { return m_mshrs.next_access(); }
    // flash invalidate all entries in cache
    void flush() { m_tag_array->flush(); }
    void invalidate() { m_tag_array->invalidate(); }
    void print(FILE *fp, unsigned &accesses, unsigned &misses) const;
    void display_state(FILE *fp) const;

    // Stat collection
    const cache_stats &get_stats() const { return m_stats; }
    unsigned get_stats(enum mem_access_type *access_type,
                       unsigned num_access_type,
                       enum cache_request_status *access_status,
                       unsigned num_access_status) const {
        return m_stats.get_stats(access_type, num_access_type, access_status,
                                 num_access_status);
    }
    void get_sub_stats(struct cache_sub_stats &css) const {
        m_stats.get_sub_stats(css);
    }
    // Clear per-window stats for AerialVision support
    void clear_pw() { m_stats.clear_pw(); }
    // Per-window sub stats for AerialVision support
    void get_sub_stats_pw(struct cache_sub_stats_pw &css) const {
        m_stats.get_sub_stats_pw(css);
    }

    // accessors for cache bandwidth availability
    bool data_port_free() const {
        return m_bandwidth_management.data_port_free();
    }
    bool fill_port_free() const {
        return m_bandwidth_management.fill_port_free();
    }
    void inc_aggregated_stats(cache_request_status status,
                              cache_request_status cache_status, mem_fetch *mf,
                              enum cache_gpu_level level);
    void inc_aggregated_fail_stats(cache_request_status status,
                                   cache_request_status cache_status,
                                   mem_fetch *mf, enum cache_gpu_level level);
    void inc_aggregated_stats_pw(cache_request_status status,
                                 cache_request_status cache_status,
                                 mem_fetch *mf, enum cache_gpu_level level);

    // This is a gapping hole we are poking in the system to quickly handle
    // filling the cache on cudamemcopies. We don't care about anything other
    // than L2 state after the memcopy - so just force the tag array to act as
    // though something is read or written without doing anything else.
    void force_tag_access(new_addr_type addr, unsigned time,
                          mem_access_sector_mask_t mask) {
        mem_access_byte_mask_t byte_mask;
        m_tag_array->fill(addr, time, mask, byte_mask, true);
    }

protected:
    // Constructor that can be used by derived classes with custom tag arrays
    baseline_cache(const char *name, cache_config &config, int core_id,
                   int type_id, mem_fetch_interface *memport,
                   enum mem_fetch_status status, tag_array *new_tag_array)
        : m_config(config), m_tag_array(new_tag_array),
          m_mshrs(config.m_mshr_entries, config.m_mshr_max_merge),
          m_bandwidth_management(config) {
        init(name, config, memport, status);
    }

protected:
    std::string m_name;

    /*不同的cache会有不同的配置文件 */
    cache_config &m_config;
    tag_array *m_tag_array;
    mshr_table m_mshrs;

    /**
     * 在baseline_cache::cycle()中，会将m_miss_queue队首的数据包mf传递给下一层内存。
     * 当遇到miss的请求需要访问下一级存储时，会把miss的请求放到m_miss_queue中。
     */
    std::list<mem_fetch *> m_miss_queue;
    enum mem_fetch_status m_miss_queue_status;

    /**
     * m_memport是cache对mem访存的接口，cache将miss请求发送至下一级存储就是通过这个接口来发送，
     * 即m_miss_queue中的数据包需要压入m_memport实现发送至下一级存储。
     */
    mem_fetch_interface *m_memport;
    cache_gpu_level m_level;
    gpgpu_sim *m_gpu;

    /**
     * 存一些memory request额外的信息，比如cache set index，方便计算
     */
    struct extra_mf_fields {
        extra_mf_fields() { m_valid = false; }
        extra_mf_fields(new_addr_type a, new_addr_type ad, unsigned i,
                        unsigned d, const cache_config &m_config) {
            m_valid = true;
            m_block_addr = a;
            m_addr = ad;
            m_cache_index = i;
            m_data_size = d;
            pending_read = m_config.m_mshr_type == SECTOR_ASSOC
                               ? m_config.m_line_sz / SECTOR_SIZE
                               : 0;
        }
        bool m_valid;
        new_addr_type m_block_addr;
        new_addr_type m_addr;
        /*在on-miss策略中，之前分配的cacheLine通过参数*/
        unsigned m_cache_index;
        unsigned m_data_size;
        // this variable is used when a load request generates multiple load
        // transactions For example, a read request from non-sector L1 request
        // sends a request to sector L2
        unsigned pending_read;
    };

    typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;

    cache_stats m_stats;

    /*判断m_miss_queue是否已满，从而判断该request能在在本周期处理，num_miss是本周期能处理的最多的miss的请求数量 */
    bool miss_queue_full(unsigned num_miss) {
        return ((m_miss_queue.size() + num_miss) >= m_config.m_miss_queue_size);
    }

    /// Read miss handler without writeback
    void send_read_request(new_addr_type addr, new_addr_type block_addr,
                           unsigned cache_index, mem_fetch *mf, unsigned time,
                           bool &do_miss, std::list<cache_event> &events,
                           bool read_only, bool wa);
    /// Read miss handler. Check MSHR hit or MSHR available
    void send_read_request(new_addr_type addr, new_addr_type block_addr,
                           unsigned cache_index, mem_fetch *mf, unsigned time,
                           bool &do_miss, bool &wb, evicted_block_info &evicted,
                           std::list<cache_event> &events, bool read_only,
                           bool wa);

    /// Sub-class containing all metadata for port bandwidth management
    class bandwidth_management {
    public:
        bandwidth_management(cache_config &config);

        /// use the data port based on the outcome and events generated by the
        /// mem_fetch request
        void use_data_port(mem_fetch *mf, enum cache_request_status outcome,
                           const std::list<cache_event> &events);

        /// use the fill port
        void use_fill_port(mem_fetch *mf);

        /// called every cache cycle to free up the ports
        void replenish_port_bandwidth();

        /// query for data port availability
        bool data_port_free() const;
        /// query for fill port availability
        bool fill_port_free() const;

    protected:
        const cache_config &m_config;

        int m_data_port_occupied_cycles; //< Number of cycle that the data port
                                         // remains used
        int m_fill_port_occupied_cycles; //< Number of cycle that the fill port
                                         // remains used
    };

    bandwidth_management m_bandwidth_management;
};

/// Read only cache
class read_only_cache : public baseline_cache {
public:
    read_only_cache(const char *name, cache_config &config, int core_id,
                    int type_id, mem_fetch_interface *memport,
                    enum mem_fetch_status status, enum cache_gpu_level level,
                    gpgpu_sim *gpu)
        : baseline_cache(name, config, core_id, type_id, memport, status, level,
                         gpu) {}

    /// Access cache for read_only_cache: returns RESERVATION_FAIL if request
    /// could not be accepted (for any reason)
    /*首先*/
    virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events);

    virtual ~read_only_cache() {}

protected:
    read_only_cache(const char *name, cache_config &config, int core_id,
                    int type_id, mem_fetch_interface *memport,
                    enum mem_fetch_status status, tag_array *new_tag_array)
        : baseline_cache(name, config, core_id, type_id, memport, status,
                         new_tag_array) {}
};

/**
 * Data cache - Implements common functions for L1 and L2 data cache
 */
class data_cache : public baseline_cache {
public:
    data_cache(const char *name, cache_config &config, int core_id, int type_id,
               mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
               enum mem_fetch_status status, mem_access_type wr_alloc_type,
               mem_access_type wrbk_type, class gpgpu_sim *gpu,
               enum cache_gpu_level level)
        : baseline_cache(name, config, core_id, type_id, memport, status, level,
                         gpu) {
        init(mfcreator);
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
        m_gpu = gpu;
    }

    virtual ~data_cache() {}

    virtual void init(mem_fetch_allocator *mfcreator) {
        m_memfetch_creator = mfcreator;

        // Set read hit function
        m_rd_hit = &data_cache::rd_hit_base;

        // Set read miss function
        m_rd_miss = &data_cache::rd_miss_base;

        // Set write hit function
        switch (m_config.m_write_policy) {
        // READ_ONLY is now a separate cache class, config is deprecated
        case READ_ONLY:
            assert(0 && "Error: Writable Data_cache set as READ_ONLY\n");
            break;
        case WRITE_BACK:
            m_wr_hit = &data_cache::wr_hit_wb;
            break;
        case WRITE_THROUGH:
            m_wr_hit = &data_cache::wr_hit_wt;
            break;
        case WRITE_EVICT:
            m_wr_hit = &data_cache::wr_hit_we;
            break;
        case LOCAL_WB_GLOBAL_WT:
            m_wr_hit = &data_cache::wr_hit_global_we_local_wb;
            break;
        default:
            assert(0 && "Error: Must set valid cache write policy\n");
            break; // Need to set a write hit function
        }

        // Set write miss function
        switch (m_config.m_write_alloc_policy) {
        case NO_WRITE_ALLOCATE:
            m_wr_miss = &data_cache::wr_miss_no_wa;
            break;
        case WRITE_ALLOCATE:
            m_wr_miss = &data_cache::wr_miss_wa_naive;
            break;
        case FETCH_ON_WRITE:
            m_wr_miss = &data_cache::wr_miss_wa_fetch_on_write;
            break;
        case LAZY_FETCH_ON_READ:
            m_wr_miss = &data_cache::wr_miss_wa_lazy_fetch_on_read;
            break;
        default:
            assert(0 && "Error: Must set valid cache write miss policy\n");
            break; // Need to set a write miss function
        }
    }

    virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events);

protected:
    data_cache(const char *name, cache_config &config, int core_id, int type_id,
               mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
               enum mem_fetch_status status, tag_array *new_tag_array,
               mem_access_type wr_alloc_type, mem_access_type wrbk_type,
               class gpgpu_sim *gpu)
        : baseline_cache(name, config, core_id, type_id, memport, status,
                         new_tag_array) {
        init(mfcreator);
        m_wr_alloc_type = wr_alloc_type;
        m_wrbk_type = wrbk_type;
        m_gpu = gpu;
    }

    /*由于L1/L2 cache在write miss / hit 采用的策略不一样 */
    mem_access_type m_wr_alloc_type; // Specifies type of write allocate request (e.g., L1 or L2)
    mem_access_type m_wrbk_type; // Specifies type of writeback request (e.g., L1 or L2)

    class gpgpu_sim *m_gpu;

    //! A general function that takes the result of a tag_array probe
    //  and performs the correspding functions based on the cache configuration
    //  The access fucntion calls this function
    enum cache_request_status
    process_tag_probe(bool wr, enum cache_request_status status,
                      new_addr_type addr, unsigned cache_index, mem_fetch *mf,
                      unsigned time, std::list<cache_event> &events);

protected:
    mem_fetch_allocator *m_memfetch_creator;

    // Functions for data cache access
    /// Sends write request to lower level memory (write or writeback)
    void send_write_request(mem_fetch *mf, cache_event request, unsigned time,
                            std::list<cache_event> &events);
    void update_m_readable(mem_fetch *mf, unsigned cache_index);
    
    // Member Function pointers - Set by configuration options
    // to the functions below each grouping
    /******* Write-hit configs *******/
    enum cache_request_status (data_cache::*m_wr_hit)(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events, enum cache_request_status status);
    /// Marks block as MODIFIED and updates block LRU
    enum cache_request_status
    wr_hit_wb(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
              unsigned time, std::list<cache_event> &events,
              enum cache_request_status status); // write-back
    enum cache_request_status
    wr_hit_wt(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
              unsigned time, std::list<cache_event> &events,
              enum cache_request_status status); // write-through

    /// Marks block as INVALID and sends write request to lower level memory
    enum cache_request_status
    wr_hit_we(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
              unsigned time, std::list<cache_event> &events,
              enum cache_request_status status); // write-evict
    enum cache_request_status wr_hit_global_we_local_wb(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events, enum cache_request_status status);
    // global write-evict, local write-back

    /******* Write-miss configs *******/
    enum cache_request_status (data_cache::*m_wr_miss)(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events, enum cache_request_status status);
    /// Sends read request, and possible write-back request,
    //  to lower level memory for a write miss with write-allocate
    enum cache_request_status
    wr_miss_wa_naive(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
                     unsigned time, std::list<cache_event> &events,
                     enum cache_request_status
                         status); // write-allocate-send-write-and-read-request
    enum cache_request_status wr_miss_wa_fetch_on_write(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events,
        enum cache_request_status
            status); // write-allocate with fetch-on-every-write
    enum cache_request_status wr_miss_wa_lazy_fetch_on_read(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events,
        enum cache_request_status
            status); // write-allocate with read-fetch-only
    enum cache_request_status wr_miss_wa_write_validate(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events,
        enum cache_request_status
            status); // write-allocate that writes with no read fetch
    enum cache_request_status
    wr_miss_no_wa(new_addr_type addr, unsigned cache_index, mem_fetch *mf,
                  unsigned time, std::list<cache_event> &events,
                  enum cache_request_status status); // no write-allocate

    // Currently no separate functions for reads
    /******* Read-hit configs *******/
    enum cache_request_status (data_cache::*m_rd_hit)(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events, enum cache_request_status status);
    enum cache_request_status rd_hit_base(new_addr_type addr,
                                          unsigned cache_index, mem_fetch *mf,
                                          unsigned time,
                                          std::list<cache_event> &events,
                                          enum cache_request_status status);

    /******* Read-miss configs *******/
    enum cache_request_status (data_cache::*m_rd_miss)(
        new_addr_type addr, unsigned cache_index, mem_fetch *mf, unsigned time,
        std::list<cache_event> &events, enum cache_request_status status);
    enum cache_request_status rd_miss_base(new_addr_type addr,
                                           unsigned cache_index, mem_fetch *mf,
                                           unsigned time,
                                           std::list<cache_event> &events,
                                           enum cache_request_status status);
};

/// This is meant to model the first level data cache in Fermi.
/// It is write-evict (global) or write-back (local) at
/// the granularity of individual blocks
/// (the policy used in fermi according to the CUDA manual)
// L1 cache采取的写策略：
//     对L1 cache写不命中时，采用write-allocate策略，将缺失块从下级存储调入L1 cache，
//                          并在L1 cache中修改。
//     对L1 cache写命中时，采用write-back策略，只写入L1 cache，不直接写入下级存储，在
//                          L1 cache的sector被逐出时才将数据写回下级缓存。
// L2 cache采取的写策略：
//     对L2 cache写不命中时，采用write-allocate策略，将缺失块从DRAM调入L2 cache，并在
//                          L2 cache中修改。
//     对L2 cache写命中时，采用write-back策略，只写入L2 cache，并不直接写入DRAM，在L2 
//                          cache的sector被逐出时才将数据写回DRAM。
class l1_cache : public data_cache {
public:
    l1_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, class gpgpu_sim *gpu,
             enum cache_gpu_level level)
        : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                     L1_WR_ALLOC_R, L1_WRBK_ACC, gpu, level) {}

    virtual ~l1_cache() {}

    virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events);

protected:
    l1_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, tag_array *new_tag_array,
             class gpgpu_sim *gpu)
        : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                     new_tag_array, L1_WR_ALLOC_R, L1_WRBK_ACC, gpu) {}
};

/// Models second level shared cache with global write-back
/// and write-allocate policies
class l2_cache : public data_cache {
public:
    l2_cache(const char *name, cache_config &config, int core_id, int type_id,
             mem_fetch_interface *memport, mem_fetch_allocator *mfcreator,
             enum mem_fetch_status status, class gpgpu_sim *gpu,
             enum cache_gpu_level level)
        : data_cache(name, config, core_id, type_id, memport, mfcreator, status,
                     L2_WR_ALLOC_R, L2_WRBK_ACC, gpu, level) {}

    virtual ~l2_cache() {}

    virtual enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                             unsigned time,
                                             std::list<cache_event> &events);
};

/*****************************************************************************/

// See the following paper to understand this cache model:
//
// Igehy, et al., Prefetching in a Texture Cache Architecture,
// Proceedings of the 1998 Eurographics/SIGGRAPH Workshop on Graphics Hardware
// http://www-graphics.stanford.edu/papers/texture_prefetch/
class tex_cache : public cache_t {
public:
    tex_cache(const char *name, cache_config &config, int core_id, int type_id,
              mem_fetch_interface *memport,
              enum mem_fetch_status request_status,
              enum mem_fetch_status rob_status)
        : m_config(config), m_tags(config, core_id, type_id),
          m_fragment_fifo(config.m_fragment_fifo_entries),
          m_request_fifo(config.m_request_fifo_entries),
          m_rob(config.m_rob_entries),
          m_result_fifo(config.m_result_fifo_entries) {
        m_name = name;
        assert(config.m_mshr_type == TEX_FIFO ||
               config.m_mshr_type == SECTOR_TEX_FIFO);
        assert(config.m_write_policy == READ_ONLY);
        assert(config.m_alloc_policy == ON_MISS);
        m_memport = memport;
        m_cache = new data_block[config.get_num_lines()];
        m_request_queue_status = request_status;
        m_rob_status = rob_status;
    }

    /// Access function for tex_cache
    /// return values: RESERVATION_FAIL if request could not be accepted
    /// otherwise returns HIT_RESERVED or MISS; NOTE: *never* returns HIT
    /// since unlike a normal CPU cache, a "HIT" in texture cache does not
    /// mean the data is ready (still need to get through fragment fifo)
    enum cache_request_status access(new_addr_type addr, mem_fetch *mf,
                                     unsigned time,
                                     std::list<cache_event> &events);
    void cycle();
    /// Place returning cache block into reorder buffer
    void fill(mem_fetch *mf, unsigned time);
    /// Are any (accepted) accesses that had to wait for memory now ready? (does
    /// not include accesses that "HIT")
    bool access_ready() const { return !m_result_fifo.empty(); }
    /// Pop next ready access (includes both accesses that "HIT" and those that
    /// "MISS")
    mem_fetch *next_access() { return m_result_fifo.pop(); }
    void display_state(FILE *fp) const;

    // accessors for cache bandwidth availability - stubs for now
    bool data_port_free() const { return true; }
    bool fill_port_free() const { return true; }

    // Stat collection
    const cache_stats &get_stats() const { return m_stats; }
    unsigned get_stats(enum mem_access_type *access_type,
                       unsigned num_access_type,
                       enum cache_request_status *access_status,
                       unsigned num_access_status) const {
        return m_stats.get_stats(access_type, num_access_type, access_status,
                                 num_access_status);
    }

    void get_sub_stats(struct cache_sub_stats &css) const {
        m_stats.get_sub_stats(css);
    }

private:
    std::string m_name;
    const cache_config &m_config;

    struct fragment_entry {
        fragment_entry() {}
        fragment_entry(mem_fetch *mf, unsigned idx, bool m, unsigned d) {
            m_request = mf;
            m_cache_index = idx;
            m_miss = m;
            m_data_size = d;
        }
        mem_fetch *m_request;   // request information
        unsigned m_cache_index; // where to look for data
        bool m_miss;            // true if sent memory request
        unsigned m_data_size;
    };

    struct rob_entry {
        rob_entry() {
            m_ready = false;
            m_time = 0;
            m_request = NULL;
        }
        rob_entry(unsigned i, mem_fetch *mf, new_addr_type a) {
            m_ready = false;
            m_index = i;
            m_time = 0;
            m_request = mf;
            m_block_addr = a;
        }
        bool m_ready;
        unsigned m_time;  // which cycle did this entry become ready?
        unsigned m_index; // where in cache should block be placed?
        mem_fetch *m_request;
        new_addr_type m_block_addr;
    };

    struct data_block {
        data_block() { m_valid = false; }
        bool m_valid;
        new_addr_type m_block_addr;
    };

    // TODO: replace fifo_pipeline with this?
    template <class T> class fifo {
    public:
        fifo(unsigned size) {
            m_size = size;
            m_num = 0;
            m_head = 0;
            m_tail = 0;
            m_data = new T[size];
        }
        bool full() const { return m_num == m_size; }
        bool empty() const { return m_num == 0; }
        unsigned size() const { return m_num; }
        unsigned capacity() const { return m_size; }
        unsigned push(const T &e) {
            assert(!full());
            m_data[m_head] = e;
            unsigned result = m_head;
            inc_head();
            return result;
        }
        T pop() {
            assert(!empty());
            T result = m_data[m_tail];
            inc_tail();
            return result;
        }
        const T &peek(unsigned index) const {
            assert(index < m_size);
            return m_data[index];
        }
        T &peek(unsigned index) {
            assert(index < m_size);
            return m_data[index];
        }
        T &peek() const { return m_data[m_tail]; }
        unsigned next_pop_index() const { return m_tail; }

    private:
        void inc_head() {
            m_head = (m_head + 1) % m_size;
            m_num++;
        }
        void inc_tail() {
            assert(m_num > 0);
            m_tail = (m_tail + 1) % m_size;
            m_num--;
        }

        unsigned m_head; // next entry goes here
        unsigned m_tail; // oldest entry found here
        unsigned m_num;  // how many in fifo?
        unsigned m_size; // maximum number of entries in fifo
        T *m_data;
    };

    tag_array m_tags;
    fifo<fragment_entry> m_fragment_fifo;
    fifo<mem_fetch *> m_request_fifo;
    fifo<rob_entry> m_rob;
    data_block *m_cache;
    fifo<mem_fetch *> m_result_fifo; // next completed texture fetch

    mem_fetch_interface *m_memport;
    enum mem_fetch_status m_request_queue_status;
    enum mem_fetch_status m_rob_status;

    struct extra_mf_fields {
        extra_mf_fields() { m_valid = false; }
        extra_mf_fields(unsigned i, const cache_config &m_config) {
            m_valid = true;
            m_rob_index = i;
            pending_read = m_config.m_mshr_type == SECTOR_TEX_FIFO
                               ? m_config.m_line_sz / SECTOR_SIZE
                               : 0;
        }
        bool m_valid;
        unsigned m_rob_index;
        unsigned pending_read;
    };

    cache_stats m_stats;

    typedef std::map<mem_fetch *, extra_mf_fields> extra_mf_fields_lookup;

    extra_mf_fields_lookup m_extra_mf_fields;
};

#endif
