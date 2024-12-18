// Copyright (c) 2009-2021, Tor M. Aamodt, Inderpreet Singh, Vijay Kandiah,
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

#ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
#define ABSTRACT_HARDWARE_MODEL_INCLUDED

// Forward declarations
class gpgpu_sim;
class kernel_info_t;
class gpgpu_context;

// Set a hard limit of 32 CTAs per shader [cuda only has 8]
#define MAX_CTA_PER_SHADER 32
#define MAX_BARRIERS_PER_CTA 16

// After expanding the vector input and output operands
#define MAX_INPUT_VALUES 24
#define MAX_OUTPUT_VALUES 8

enum _memory_space_t {
    undefined_space = 0,
    reg_space,
    local_space,
    shared_space,
    sstarr_space,
    param_space_unclassified,
    param_space_kernel, /* global to all threads in a kernel : read-only */
    param_space_local,  /* local to a thread : read-writable */
    const_space,
    tex_space,
    surf_space,
    global_space,
    generic_space,
    instruction_space
};

#ifndef COEFF_STRUCT
#define COEFF_STRUCT

struct PowerscalingCoefficients {
    double int_coeff;
    double int_mul_coeff;
    double int_mul24_coeff;
    double int_mul32_coeff;
    double int_div_coeff;
    double fp_coeff;
    double dp_coeff;
    double fp_mul_coeff;
    double fp_div_coeff;
    double dp_mul_coeff;
    double dp_div_coeff;
    double sqrt_coeff;
    double log_coeff;
    double sin_coeff;
    double exp_coeff;
    double tensor_coeff;
    double tex_coeff;
};
#endif

enum FuncCache {
    FuncCachePreferNone = 0,
    FuncCachePreferShared = 1,
    FuncCachePreferL1 = 2
};

enum AdaptiveCache { FIXED = 0, ADAPTIVE_CACHE = 1 };

#ifdef __cplusplus

#include <set>
#include <stdio.h>
#include <string.h>

typedef unsigned long long new_addr_type;
typedef unsigned long long cudaTextureObject_t;
typedef unsigned long long address_type;
typedef unsigned long long addr_t;

// the following are operations the timing model can see
#define SPECIALIZED_UNIT_NUM 8
#define SPEC_UNIT_START_ID 100

enum uarch_op_t {
    NO_OP = -1,
    ALU_OP = 1,
    SFU_OP,
    TENSOR_CORE_OP,
    DP_OP,
    SP_OP,
    INTP_OP,
    ALU_SFU_OP,
    LOAD_OP,
    TENSOR_CORE_LOAD_OP,
    TENSOR_CORE_STORE_OP,
    STORE_OP,
    BRANCH_OP,
    BARRIER_OP,
    MEMORY_BARRIER_OP,
    CALL_OPS,
    RET_OPS,
    EXIT_OPS,
    SPECIALIZED_UNIT_1_OP = SPEC_UNIT_START_ID,
    SPECIALIZED_UNIT_2_OP,
    SPECIALIZED_UNIT_3_OP,
    SPECIALIZED_UNIT_4_OP,
    SPECIALIZED_UNIT_5_OP,
    SPECIALIZED_UNIT_6_OP,
    SPECIALIZED_UNIT_7_OP,
    SPECIALIZED_UNIT_8_OP
};
typedef enum uarch_op_t op_type;

enum uarch_bar_t { NOT_BAR = -1, SYNC = 1, ARRIVE, RED };
typedef enum uarch_bar_t barrier_type;

enum uarch_red_t { NOT_RED = -1, POPC_RED = 1, AND_RED, OR_RED };
typedef enum uarch_red_t reduction_type;

enum uarch_operand_type_t { UN_OP = -1, INT_OP, FP_OP };
typedef enum uarch_operand_type_t types_of_operands;

enum special_operations_t {
    OTHER_OP,
    INT__OP,
    INT_MUL24_OP,
    INT_MUL32_OP,
    INT_MUL_OP,
    INT_DIV_OP,
    FP_MUL_OP,
    FP_DIV_OP,
    FP__OP,
    FP_SQRT_OP,
    FP_LG_OP,
    FP_SIN_OP,
    FP_EXP_OP,
    DP_MUL_OP,
    DP_DIV_OP,
    DP___OP,
    TENSOR__OP,
    TEX__OP
};

typedef enum special_operations_t
    special_ops; // Required to identify for the power model
enum operation_pipeline_t {
    UNKOWN_OP,
    SP__OP,
    DP__OP,
    INTP__OP,
    SFU__OP,
    TENSOR_CORE__OP,
    MEM__OP,
    SPECIALIZED__OP,
};
typedef enum operation_pipeline_t operation_pipeline;
enum mem_operation_t { NOT_TEX, TEX };
typedef enum mem_operation_t mem_operation;

enum _memory_op_t { no_memory_op = 0, memory_load, memory_store };

#include <algorithm>
#include <assert.h>
#include <bitset>
#include <deque>
#include <list>
#include <map>
#include <stdlib.h>
#include <vector>

#if !defined(__VECTOR_TYPES_H__)
#include "vector_types.h"
#endif
struct dim3comp {
    bool operator()(const dim3 &a, const dim3 &b) const {
        if (a.z < b.z)
            return true;
        else if (a.y < b.y)
            return true;
        else if (a.x < b.x)
            return true;
        else
            return false;
    }
};

/**
 * 实现在gpu-sim.cc中，实现传入的dim3维度依次在x，y，z维度的加1
 */
void increment_x_then_y_then_z(dim3 &i, const dim3 &bound);

// Jin: child kernel information for CDP
#include "stream_manager.h"
class stream_manager;
struct CUstream_st;
// extern stream_manager * g_stream_manager;
// support for pinned memories added
extern std::map<void *, void **> pinned_memory;
extern std::map<void *, size_t> pinned_memory_size;

/**
 * 包含kernel函数如下信息：
 *  - Grid/Block dim的信息
 *  - the function_info object associated with the kernel entry point
 *  - launch status
 *  - param memory： memory allocated for the kernel arguments in param memory
 *  - active threads inside that kernel (ptx_thread_info).
 */
class kernel_info_t {
public:
    //   kernel_info_t()
    //   {
    //      m_valid=false;
    //      m_kernel_entry=NULL;
    //      m_uid=0;
    //      m_num_cores_running=0;
    //      m_param_mem=NULL;
    //   }

    /**暂时用不到该构造函数 */
    kernel_info_t(dim3 gridDim, dim3 blockDim, class function_info *entry,
                  unsigned long long streamID);
    
    /*主要使用的构造函数 */
    kernel_info_t(
        dim3 gridDim, dim3 blockDim, class function_info *entry,
        std::map<std::string, const struct cudaArray *> nameToCudaArray,
        std::map<std::string, const struct textureInfo *> nameToTextureInfo);
    ~kernel_info_t();

    /*运行当前kernel的simt core的数量加一，在issue_block2core()中调用 */
    void inc_running() { m_num_cores_running++; }

    /*运行当前kernel的simt core的数量减一*/
    void dec_running() {
        assert(m_num_cores_running > 0);
        m_num_cores_running--;
    }
    bool running() const { return m_num_cores_running > 0; }
    bool done() const { return no_more_ctas_to_run() && !running(); }
    class function_info *entry() { return m_kernel_entry; }
    const class function_info *entry() const { return m_kernel_entry; }

    /*返回grid中所有cta的数量 */
    size_t num_blocks() const {
        return m_grid_dim.x * m_grid_dim.y * m_grid_dim.z;
    }

    /*返回每个cta中thread的数量 */
    size_t threads_per_cta() const {
        return m_block_dim.x * m_block_dim.y * m_block_dim.z;
    }

    dim3 get_grid_dim() const { return m_grid_dim; }
    dim3 get_cta_dim() const { return m_block_dim; }

    void increment_cta_id() {
        increment_x_then_y_then_z(m_next_cta, m_grid_dim);
        m_next_tid.x = 0;
        m_next_tid.y = 0;
        m_next_tid.z = 0;
    }
    
    /*得到kernel要执行的下一个CTA的坐标 */
    dim3 get_next_cta_id() const { return m_next_cta; }

    /*获取下一个要发射的CTA的索引。CTA的全局索引与CUDA编程模型中的线程块索引类似，其ID算法如下*/
    unsigned get_next_cta_id_single() const {
        return m_next_cta.x + m_grid_dim.x * m_next_cta.y +
               m_grid_dim.x * m_grid_dim.y * m_next_cta.z;
    }

    /**
     * m_next_cta是用于标识下一个要发射的CTA的坐标，它的值是一个全局ID，属于dim3类型，具有.x/.y/.z三个分值。
     * GPU硬件配置的CTA的全局ID的范围为：
        m_next_cta.x < m_grid_dim.x &&
        m_next_cta.y < m_grid_dim.y &&
        m_next_cta.z < m_grid_dim.z
    因此如果标识下一个要发射的CTA的全局ID的任意一维超过CUDA代码设置的Grid的对应范围，就代表内核函
    数上已经没有CTA可执行，内核函数的所有CTA均已经执行完毕
     */
    bool no_more_ctas_to_run() const {
        return (m_next_cta.x >= m_grid_dim.x || m_next_cta.y >= m_grid_dim.y ||
                m_next_cta.z >= m_grid_dim.z);
    }

    /**
     * CTA内线程的坐标按照x，y，z的次序移动
    */
    void increment_thread_id() {
        increment_x_then_y_then_z(m_next_tid, m_block_dim);
    }
    
    /*kernel要执行的下一个thread在CTA内的坐标 */
    dim3 get_next_thread_id_3d() const { return m_next_tid; }

    /*kernel要执行的下一个thread在CTA内的索引*/
    unsigned get_next_thread_id() const {
        return m_next_tid.x + m_block_dim.x * m_next_tid.y +
               m_block_dim.x * m_block_dim.y * m_next_tid.z;
    }

    /*还有剩余的CTA等待function模拟 */
    bool more_threads_in_cta() const {
        return m_next_tid.z < m_block_dim.z && m_next_tid.y < m_block_dim.y &&
               m_next_tid.x < m_block_dim.x;
    }
    unsigned get_uid() const { return m_uid; }
    unsigned long long get_streamID() const { return m_streamID; }
    std::string get_name() const { return name(); }
    std::string name() const;

    std::list<class ptx_thread_info *> &active_threads() {
        return m_active_threads;
    }
    class memory_space *get_param_memory() { return m_param_mem; }

    // The following functions access texture bindings present at the kernel's
    // launch

    const struct cudaArray *get_texarray(const std::string &texname) const {
        std::map<std::string, const struct cudaArray *>::const_iterator t =
            m_NameToCudaArray.find(texname);
        assert(t != m_NameToCudaArray.end());
        return t->second;
    }

    const struct textureInfo *get_texinfo(const std::string &texname) const {
        std::map<std::string, const struct textureInfo *>::const_iterator t =
            m_NameToTextureInfo.find(texname);
        assert(t != m_NameToTextureInfo.end());
        return t->second;
    }

private:
    /**把copy constructor和copy operator放在private即可禁用拷贝构造和拷贝赋值 */
    kernel_info_t(const kernel_info_t &);  // disable copy constructor
    void operator=(const kernel_info_t &); // disable copy operator

    /*kernel的entry point */
    class function_info *m_kernel_entry;

    /*kernel_info_t对象的唯一标识符，用于放进m_finished_kernel中*/
    unsigned m_uid; // Kernel ID

    /*该kernel对应的cuda stream的uid */
    unsigned long long m_streamID;

    // These maps contain the snapshot of the texture mappings at kernel launch
    std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
    std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;

    /*grid和grid的维度 */
    dim3 m_grid_dim;
    dim3 m_block_dim;

    /*下一个要执行的CTA坐标 */
    dim3 m_next_cta;
    dim3 m_next_tid;

    /*正在执行该kernel的simt core的数量，构造函数中初始化为0，通过*/
    unsigned m_num_cores_running;

    /*该kernel被function  sim中对应的活跃thread信息 */
    std::list<class ptx_thread_info *> m_active_threads;

    class memory_space *m_param_mem;

public:
    // Jin: parent and child kernel management for CDP
    void set_parent(kernel_info_t *parent, dim3 parent_ctaid, dim3 parent_tid);
    void set_child(kernel_info_t *child);
    void remove_child(kernel_info_t *child);
    bool is_finished();
    bool children_all_finished();
    void notify_parent_finished();
    CUstream_st *create_stream_cta(dim3 ctaid);
    CUstream_st *get_default_stream_cta(dim3 ctaid);
    bool cta_has_stream(dim3 ctaid, CUstream_st *stream);
    void destroy_cta_streams();
    void print_parent_info();
    kernel_info_t *get_parent() { return m_parent_kernel; }

private:
    kernel_info_t *m_parent_kernel;
    dim3 m_parent_ctaid;
    dim3 m_parent_tid;
    std::list<kernel_info_t *> m_child_kernels; // child kernel launched
    std::map<dim3, std::list<CUstream_st *>, dim3comp>
        m_cta_streams; // streams created in each CTA

    // Jin: kernel timing
public:
    unsigned long long launch_cycle;
    unsigned long long start_cycle;

    /*该kernel对应的结束周期，为gpu_sim_cycle + gpu_tot_sim_cycle */
    unsigned long long end_cycle;
    unsigned m_launch_latency;

    mutable bool cache_config_set;

    /*this used for any CPU-GPU kernel latency and counted in the gpu_cycle */
    /*m_kernel_TB_latency是GPGPU-Sim中用于表示内核启动时间的变量，只有为0表示该kernel可以被gpgpu-sim仿真 */
    unsigned m_kernel_TB_latency;
};

class core_config {
public:
    core_config(gpgpu_context *ctx) {
        gpgpu_ctx = ctx;
        m_valid = false;
        num_shmem_bank = 16;
        shmem_limited_broadcast = false;
        gpgpu_shmem_sizeDefault = (unsigned)-1;
        gpgpu_shmem_sizePrefL1 = (unsigned)-1;
        gpgpu_shmem_sizePrefShared = (unsigned)-1;
    }
    virtual void init() = 0;

    bool m_valid;
    unsigned warp_size;
    // backward pointer
    class gpgpu_context *gpgpu_ctx;

    // off-chip memory request architecture parameters
    int gpgpu_coalesce_arch;

    // shared memory bank conflict checking parameters
    // Limit shared memory to do one broadcast per cycle (default on)
    bool shmem_limited_broadcast;
    static const address_type WORD_SIZE = 4;
    unsigned num_shmem_bank;
    unsigned shmem_bank_func(address_type addr) const {
        return ((addr / WORD_SIZE) % num_shmem_bank);
    }
    /*Number of portions a warp is divided into for shared memory bank conflict check */
    unsigned mem_warp_parts;
    mutable unsigned gpgpu_shmem_size;
    char *gpgpu_shmem_option;
    std::vector<unsigned> shmem_opt_list;
    unsigned gpgpu_shmem_sizeDefault;
    unsigned gpgpu_shmem_sizePrefL1;
    unsigned gpgpu_shmem_sizePrefShared;
    unsigned mem_unit_ports;

    // texture and constant cache line sizes (used to determine number of memory
    // accesses)
    unsigned gpgpu_cache_texl1_linesize;
    unsigned gpgpu_cache_constl1_linesize;

    unsigned gpgpu_max_insn_issue_per_warp;
    bool gmem_skip_L1D; // on = global memory access always skip the L1 cache

    bool adaptive_cache_config;
};

// bounded stack that implements simt reconvergence using pdom mechanism from
// MICRO'07 paper
const unsigned MAX_WARP_SIZE = 32;
typedef std::bitset<MAX_WARP_SIZE> active_mask_t;
#define MAX_WARP_SIZE_SIMT_STACK MAX_WARP_SIZE
typedef std::bitset<MAX_WARP_SIZE_SIMT_STACK> simt_mask_t;
typedef std::vector<address_type> addr_vector_t;

/**
 * 每一个warp都拥有一个simt堆栈，gpgpu-sim中只有在issue阶段会和simt堆栈产生交互，主要是：
 * 1. 在scheduler_unit::cycle()中调用 get_active_mask()
 * 2. 在shader_core_ctx::issue_warp()通过core_t::updateSIMTStack()更新simt堆栈
 */
class simt_stack {
public:
    simt_stack(unsigned wid, unsigned warpSize, class gpgpu_sim *gpu);

    /*清空 m_stack里的所有entry*/
    void reset();
    void launch(address_type start_pc, const simt_mask_t &active_mask);
    void update(simt_mask_t &thread_done, addr_vector_t &next_pc,
                address_type recvg_pc, op_type next_inst_op,
                unsigned next_inst_size, address_type next_inst_pc);

    const simt_mask_t &get_active_mask() const;
    void get_pdom_stack_top_info(unsigned *pc, unsigned *rpc) const;
    unsigned get_rp() const;
    void print(FILE *fp) const;
    void resume(char *fname);
    void print_checkpoint(FILE *fout) const;

protected:
    unsigned m_warp_id;
    unsigned m_warp_size;

    enum stack_entry_type {
        STACK_ENTRY_TYPE_NORMAL = 0,
        STACK_ENTRY_TYPE_CALL
    };

    struct simt_stack_entry {
        /*下一条需要被执行指令的PC（Next PC，NPC），为该分支内需要执行的指令PC*/
        address_type m_pc;
        unsigned int m_calldepth;
        /*代表该指令的活跃掩码*/
        simt_mask_t m_active_mask;
        /*分支重聚点的PC（Reconvergence PC，RPC），是直接后必经结点的PC（IPDOM*/
        address_type m_recvg_pc;
        /*发生分支的时刻，时钟周期*/
        unsigned long long m_branch_div_cycle;
        stack_entry_type m_type;
        simt_stack_entry()
            : m_pc(-1), m_calldepth(0), m_active_mask(), m_recvg_pc(-1),
              m_branch_div_cycle(0), m_type(STACK_ENTRY_TYPE_NORMAL) {};
    };

    std::deque<simt_stack_entry> m_stack;

    class gpgpu_sim *m_gpu;
};

// Let's just upgrade to C++11 so we can use constexpr here...
// start allocating from this address (lower values used for allocating globals
// in .ptx file)
const unsigned long long GLOBAL_HEAP_START = 0xC0000000;
// Volta max shmem size is 96kB
const unsigned long long SHARED_MEM_SIZE_MAX = 96 * (1 << 10);
// Volta max local mem is 16kB
const unsigned long long LOCAL_MEM_SIZE_MAX = 1 << 14;
// Volta Titan V has 80 SMs
const unsigned MAX_STREAMING_MULTIPROCESSORS = 80;
// Max 2048 threads / SM
const unsigned MAX_THREAD_PER_SM = 1 << 11;
// MAX 64 warps / SM
const unsigned MAX_WARP_PER_SM = 1 << 6;
const unsigned long long TOTAL_LOCAL_MEM_PER_SM =
    MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long TOTAL_SHARED_MEM =
    MAX_STREAMING_MULTIPROCESSORS * SHARED_MEM_SIZE_MAX;
const unsigned long long TOTAL_LOCAL_MEM =
    MAX_STREAMING_MULTIPROCESSORS * MAX_THREAD_PER_SM * LOCAL_MEM_SIZE_MAX;
const unsigned long long SHARED_GENERIC_START =
    GLOBAL_HEAP_START - TOTAL_SHARED_MEM;
const unsigned long long LOCAL_GENERIC_START =
    SHARED_GENERIC_START - TOTAL_LOCAL_MEM;
const unsigned long long STATIC_ALLOC_LIMIT =
    GLOBAL_HEAP_START - (TOTAL_LOCAL_MEM + TOTAL_SHARED_MEM);

#if !defined(__CUDA_RUNTIME_API_H__)

#include "builtin_types.h"

struct cudaArray {
    void *devPtr;
    int devPtr32;
    struct cudaChannelFormatDesc desc;
    int width;
    int height;
    int size; // in bytes
    unsigned dimensions;
};

#endif

// Struct that record other attributes in the textureReference declaration
// - These attributes are passed thru __cudaRegisterTexture()
struct textureReferenceAttr {
    const struct textureReference *m_texref;
    int m_dim;
    enum cudaTextureReadMode m_readmode;
    int m_ext;
    textureReferenceAttr(const struct textureReference *texref, int dim,
                         enum cudaTextureReadMode readmode, int ext)
        : m_texref(texref), m_dim(dim), m_readmode(readmode), m_ext(ext) {}
};

class gpgpu_functional_sim_config {
public:
    void reg_options(class OptionParser *opp);

    void ptx_set_tex_cache_linesize(unsigned linesize);

    unsigned get_forced_max_capability() const {
        return m_ptx_force_max_capability;
    }
    bool convert_to_ptxplus() const { return m_ptx_convert_to_ptxplus; }
    bool use_cuobjdump() const { return m_ptx_use_cuobjdump; }
    bool experimental_lib_support() const { return m_experimental_lib_support; }

    int get_ptx_inst_debug_to_file() const { return g_ptx_inst_debug_to_file; }
    const char *get_ptx_inst_debug_file() const {
        return g_ptx_inst_debug_file;
    }
    int get_ptx_inst_debug_thread_uid() const {
        return g_ptx_inst_debug_thread_uid;
    }
    unsigned get_texcache_linesize() const { return m_texcache_linesize; }
    int get_checkpoint_option() const { return checkpoint_option; }
    int get_checkpoint_kernel() const { return checkpoint_kernel; }
    int get_checkpoint_CTA() const { return checkpoint_CTA; }
    int get_resume_option() const { return resume_option; }
    int get_resume_kernel() const { return resume_kernel; }
    int get_resume_CTA() const { return resume_CTA; }
    int get_checkpoint_CTA_t() const { return checkpoint_CTA_t; }
    int get_checkpoint_insn_Y() const { return checkpoint_insn_Y; }

private:
    // PTX options
    int m_ptx_convert_to_ptxplus;

    /*cuda4.0后使用cuobjdump去解析PTX,SASS等指令 */
    int m_ptx_use_cuobjdump;
    int m_experimental_lib_support;
    unsigned m_ptx_force_max_capability;
    int checkpoint_option;
    int checkpoint_kernel;
    int checkpoint_CTA;
    unsigned resume_option;
    unsigned resume_kernel;
    unsigned resume_CTA;
    unsigned checkpoint_CTA_t;
    int checkpoint_insn_Y;
    int g_ptx_inst_debug_to_file;
    char *g_ptx_inst_debug_file;
    int g_ptx_inst_debug_thread_uid;

    unsigned m_texcache_linesize;
};

/**
 * 实现 functional GPU simulator的最顶层的类， Class gpgpu_sim (the top-level
 * GPU timing simulation model) 会继承该类 拥有actual buffer that implements
 * global/texture memory spaces
 */
class gpgpu_t {
public:
    gpgpu_t(const gpgpu_functional_sim_config &config, gpgpu_context *ctx);
    // backward pointer

    /*全局唯一的gpgpu */
    class gpgpu_context *gpgpu_ctx;
    int checkpoint_option;
    int checkpoint_kernel;
    int checkpoint_CTA;
    unsigned resume_option;
    unsigned resume_kernel;
    unsigned resume_CTA;
    unsigned checkpoint_CTA_t;
    int checkpoint_insn_Y;

    // Move some cycle core stats here instead of being global
    /*整个时序模拟经历的cycle数,每次执行一次gpgpu_sim::cycle()其值+1*/
    /**
     * 可以通过在gdb中使用 g_single_step=10
     * 准确的停留在gpu_sim_cycle=10的状态,从而查询各个部件的信息
     */
    unsigned long long gpu_sim_cycle;

    /*gpu_tot_sim_cycle是执行当前阶段之前的所有前绪指令的延迟, 比如memcpy等 */
    unsigned long long gpu_tot_sim_cycle;

    /**
     * 下面用来建模GPU memory managements the simulated GPU memory space
     * (malloc, memcpy, texture-bindin, ...). 这些函数均在cuda-sim.cc文件中实现
     * 这些函数被CUDA/OpenCL API implementations实现调用
     */
    void *gpu_malloc(size_t size);
    void *gpu_mallocarray(size_t count);
    void gpu_memset(size_t dst_start_addr, int c, size_t count);
    void memcpy_to_gpu(size_t dst_start_addr, const void *src, size_t count);
    void memcpy_from_gpu(void *dst, size_t src_start_addr, size_t count);
    void memcpy_gpu_to_gpu(size_t dst, size_t src, size_t count);

    class memory_space *get_global_memory() { return m_global_mem; }
    class memory_space *get_tex_memory() { return m_tex_mem; }
    class memory_space *get_surf_memory() { return m_surf_mem; }

    void gpgpu_ptx_sim_bindTextureToArray(const struct textureReference *texref,
                                          const struct cudaArray *array);
    void gpgpu_ptx_sim_bindNameToTexture(const char *name,
                                         const struct textureReference *texref,
                                         int dim, int readmode, int ext);
    void gpgpu_ptx_sim_unbindTexture(const struct textureReference *texref);
    const char *
    gpgpu_ptx_sim_findNamefromTexture(const struct textureReference *texref);

    const struct textureReference *
    get_texref(const std::string &texname) const {
        std::map<std::string,
                 std::set<const struct textureReference *>>::const_iterator t =
            m_NameToTextureRef.find(texname);
        assert(t != m_NameToTextureRef.end());
        return *(t->second.begin());
    }

    const struct cudaArray *get_texarray(const std::string &texname) const {
        std::map<std::string, const struct cudaArray *>::const_iterator t =
            m_NameToCudaArray.find(texname);
        assert(t != m_NameToCudaArray.end());
        return t->second;
    }

    const struct textureInfo *get_texinfo(const std::string &texname) const {
        std::map<std::string, const struct textureInfo *>::const_iterator t =
            m_NameToTextureInfo.find(texname);
        assert(t != m_NameToTextureInfo.end());
        return t->second;
    }

    const struct textureReferenceAttr *
    get_texattr(const std::string &texname) const {
        std::map<std::string,
                 const struct textureReferenceAttr *>::const_iterator t =
            m_NameToAttribute.find(texname);
        assert(t != m_NameToAttribute.end());
        return t->second;
    }

    const gpgpu_functional_sim_config &get_config() const {
        return m_function_model_config;
    }
    FILE *get_ptx_inst_debug_file() { return ptx_inst_debug_file; }

    //  These maps return the current texture mappings for the GPU at any given
    //  time.
    std::map<std::string, const struct cudaArray *> getNameArrayMapping() {
        return m_NameToCudaArray;
    }
    std::map<std::string, const struct textureInfo *> getNameInfoMapping() {
        return m_NameToTextureInfo;
    }

    virtual ~gpgpu_t() {}

protected:
    /*用来解析function sim的配置文件 */
    const gpgpu_functional_sim_config &m_function_model_config;
    FILE *ptx_inst_debug_file;

    /*模拟global memory*/
    class memory_space *m_global_mem;
    class memory_space *m_tex_mem;
    class memory_space *m_surf_mem;

    /*下一个内存分配的起始地址，每次分配完加上size，并且确保对齐 */
    unsigned long long m_dev_malloc;
    //  These maps contain the current texture mappings for the GPU at any given
    //  time.
    std::map<std::string, std::set<const struct textureReference *>>
        m_NameToTextureRef;
    std::map<const struct textureReference *, std::string> m_TextureRefToName;
    std::map<std::string, const struct cudaArray *> m_NameToCudaArray;
    std::map<std::string, const struct textureInfo *> m_NameToTextureInfo;
    std::map<std::string, const struct textureReferenceAttr *>
        m_NameToAttribute;
};

/**
 * kernel 的PTX分析出的一些参数
 * Holds properties of the kernel (Kernel's resource use).
 * These will be set to zero if a ptxinfo file is not present.
 */
struct gpgpu_ptx_sim_info {
    /*local memory大小 */
    int lmem;
    /*shared memory大小 */
    int smem;
    /*constant memory大小 */
    int cmem;
    /*global memory大小 */
    int gmem;
    /*寄存器数量 */
    int regs;
    /*最大线程数 */
    unsigned maxthreads;
    /*PTX version */
    unsigned ptx_version;
    /*目标 SM id */
    unsigned sm_target;
};

struct gpgpu_ptx_sim_arg {
    gpgpu_ptx_sim_arg() { m_start = NULL; }
    gpgpu_ptx_sim_arg(const void *arg, size_t size, size_t offset) {
        m_start = arg;
        m_nbytes = size;
        m_offset = offset;
    }
    const void *m_start;
    size_t m_nbytes;
    size_t m_offset;
};

typedef std::list<gpgpu_ptx_sim_arg> gpgpu_ptx_sim_arg_list_t;

class memory_space_t {
public:
    memory_space_t() {
        m_type = undefined_space;
        m_bank = 0;
    }
    memory_space_t(const enum _memory_space_t &from) {
        m_type = from;
        m_bank = 0;
    }
    bool operator==(const memory_space_t &x) const {
        return (m_bank == x.m_bank) && (m_type == x.m_type);
    }
    bool operator!=(const memory_space_t &x) const { return !(*this == x); }
    bool operator<(const memory_space_t &x) const {
        if (m_type < x.m_type)
            return true;
        else if (m_type > x.m_type)
            return false;
        else if (m_bank < x.m_bank)
            return true;
        return false;
    }
    enum _memory_space_t get_type() const { return m_type; }
    void set_type(enum _memory_space_t t) { m_type = t; }
    unsigned get_bank() const { return m_bank; }
    void set_bank(unsigned b) { m_bank = b; }
    bool is_const() const {
        return (m_type == const_space) || (m_type == param_space_kernel);
    }
    bool is_local() const {
        return (m_type == local_space) || (m_type == param_space_local);
    }
    bool is_global() const { return (m_type == global_space); }

private:
    enum _memory_space_t m_type;
    unsigned m_bank; // n in ".const[n]"; note .const == .const[0] (see PTX 2.1
                     // manual, sec. 5.1.3)
};

/*用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，即每次访存最大数据 128 字节。*/
const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
/*一个 cache line 包含 SECTOR_CHUNCK_SIZE = 4 个 sectors*/
const unsigned SECTOR_CHUNCK_SIZE = 4; // four sectors
const unsigned SECTOR_SIZE = 32;       // sector is 32 bytes width
/*用于标记一次访存操作中的 sector 掩码，4 个 sector，每个 sector 有 32 个字节数据*/
typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
#define NO_PARTIAL_WRITE (mem_access_byte_mask_t())

/**
 * 下面复杂的宏展开主要是实现了如下功能：
 * enum mem_access_type {
    GLOBAL_ACC_R,  // 从 global memory 读
    LOCAL_ACC_R,   // 从 local memory 读
    CONST_ACC_R,   // 从常量缓存读
    TEXTURE_ACC_R, // 从纹理缓存读
    GLOBAL_ACC_W,  // 向 global memory 写
    LOCAL_ACC_W,   // 向 local memory 写
    L1_WRBK_ACC,   // L1缓存write back
    L2_WRBK_ACC,   // 当 L2 cache 写不命中时，采取 lazy_fetch_on_read 策略，当找到一个 cache 
                   // block 逐出时，如果这个 cache block 是被 MODIFIED，则需要将这个 cache block 写回到
    INST_ACC_R,    // 从指令缓存读
    L1_WR_ALLOC_R,
    L2_WR_ALLOC_R,
};
 */
#define MEM_ACCESS_TYPE_TUP_DEF                                                \
    MA_TUP_BEGIN(mem_access_type)                                              \
    MA_TUP(GLOBAL_ACC_R), MA_TUP(LOCAL_ACC_R), MA_TUP(CONST_ACC_R),            \
        MA_TUP(TEXTURE_ACC_R), MA_TUP(GLOBAL_ACC_W), MA_TUP(LOCAL_ACC_W),      \
        MA_TUP(L1_WRBK_ACC), MA_TUP(L2_WRBK_ACC), MA_TUP(INST_ACC_R),          \
        MA_TUP(L1_WR_ALLOC_R), MA_TUP(L2_WR_ALLOC_R),                          \
        MA_TUP(NUM_MEM_ACCESS_TYPE) MA_TUP_END(mem_access_type)

#define MA_TUP_BEGIN(X) enum X {
#define MA_TUP(X) X
#define MA_TUP_END(X)                                                          \
    }                                                                          \
    ;
MEM_ACCESS_TYPE_TUP_DEF
#undef MA_TUP_BEGIN
#undef MA_TUP
#undef MA_TUP_END

const char *mem_access_type_str(enum mem_access_type access_type);

/*
缓存运算符的类型。
PTX ISA 2.0 版在加载和存储指令上引入了可选的缓存运算符。缓存运算符需要 sm_20 或更高的
目标体系结构。加载或存储指令上的缓存运算符仅被视为性能提示。对 ld 或 st 指令使用缓存运
算符不会改变程序的内存一致性行为。对于 sm_20 及更高版本，缓存运算符具有以下定义和行为：
1. 内存 Load 指令的缓存运算符：
 .ca: Cache at all levels，所有级别的缓存，可能会再次访问。
      默认的 Load 指令缓存操作是 ld.ca，它使用正常的逐出策略在所有级别（L1 和 L2）中
      分配缓存线。全局数据在 L2 级是一致的，但多个 L1 缓存对于全局数据来说是不一致的。
      如果一个线程通过一个 L1 缓存存储到全局内存，而第二个线程通过第二个 L1 缓存加载该
      地址，并使用 ld.ca，则第二个可能会得到过时的 L1 缓存数据，而不是第一个线程存储的
      数据。驱动程序必须使并行线程的从属网格之间的全局 L1 缓存线无效。然后，第一个网格
      程序的存储由第二个网格程序正确获取，该程序发出 L1 中缓存的默认 ld.ca 加载。
 .cg  Cache at global level，全局级缓存（L2 及以下缓存，而不是 L1）。
      使用 ld.cg 全局性地加载，绕过 L1 缓存，并仅缓存在 L2 缓存中。
 .cs  Cache streaming，缓存流，可能会被访问一次。
      ld.cs 加载缓存的流操作在 L1 和 L2 分配全局行，采用驱逐优先的策略在 L1 和 L2 分
      配全局行，以限制临时流数据对缓存污染，这些数据可能被访问一次或两次。当 ld.cs 被
      应用到一个本地窗口地址时，它执行 ld.lu 操作。
 .lu  Last use。
      编译器/程序员在恢复溢出的寄存器和弹出函数堆栈框架时可以使用 ld.lu，以避免不必要
      的写回不会再使用的行。ld.lu 指令在全局地址上执行一个加载缓存的流操作（ld.cs）。
 .cv  不要再次缓存和获取（考虑缓存的系统内存线已过时，再次获取）。应用于全局系统内存地
      址的 ld.cv 加载操作使匹配的 L2 行无效（丢弃），并在每次新加载时重新获取该行。
2. 内存 Store 指令的缓存运算符：
 .wb  Cache write-back all coherent levels，缓存写回所有级别一致。
      默认的存储指令缓存操作是 st.wb，它使用正常的逐出策略写回一致缓存级别的缓存行。如
      果一个线程绕过其 L1 缓存存储到全局内存，而另一个 SM 中的第二个线程稍后通过具有 
      ld.ca 的不同 L1 缓存从该地址加载，则第二个可能会命中过时的 L1 缓存数据，而不是从
      第一个线程存储的 L2 或内存中获取数据。驱动程序必须使线程阵列的从属网格之间的全局 
      L1 缓存线无效。然后，第一个网格程序的存储在 L1 中正确丢失，并由发出默认 ld.ca 加
      载的第二个网格程序获取。
 .wt  Cache write-through (to system memory).
      st.wt 存储写入操作应用于通过二级缓存写入的全局系统内存地址。
*/
enum cache_operator_type {
    CACHE_UNDEFINED,

    // loads
    CACHE_ALL,      // .ca
    CACHE_LAST_USE, // .lu
    CACHE_VOLATILE, // .cv
    CACHE_L1,       // .nc

    // loads and stores
    CACHE_STREAMING, // .cs
    CACHE_GLOBAL,    // .cg

    // stores
    CACHE_WRITE_BACK,   // .wb
    CACHE_WRITE_THROUGH // .wt
};

/**
 *  mem_access_t用于建模wrap发起的memory request，其主要功能为:
 *   - Coalesced memory access from a warp
 *   - Consumed in ldst_unit
    该类常为类 mem_fetch 的参数，在每一次访存中进行实例化
 */
class mem_access_t {
public:
    mem_access_t(gpgpu_context *ctx) { init(ctx); }
    mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
                 bool wr, gpgpu_context *ctx) {
        init(ctx);
        m_type = type;
        m_addr = address;
        m_req_size = size;
        m_write = wr;
    }
    mem_access_t(mem_access_type type, new_addr_type address, unsigned size,
                 bool wr, const active_mask_t &active_mask,
                 const mem_access_byte_mask_t &byte_mask,
                 const mem_access_sector_mask_t &sector_mask,
                 gpgpu_context *ctx)
        : m_warp_mask(active_mask), m_byte_mask(byte_mask),
          m_sector_mask(sector_mask) {
        init(ctx);
        m_type = type;
        m_addr = address;
        m_req_size = size;
        m_write = wr;
    }

    new_addr_type get_addr() const { return m_addr; }
    void set_addr(new_addr_type addr) { m_addr = addr; }
    unsigned get_size() const { return m_req_size; }
    const active_mask_t &get_warp_mask() const { return m_warp_mask; }
    bool is_write() const { return m_write; }
    enum mem_access_type get_type() const { return m_type; }
    mem_access_byte_mask_t get_byte_mask() const { return m_byte_mask; }
    mem_access_sector_mask_t get_sector_mask() const { return m_sector_mask; }

    void print(FILE *fp) const {
        fprintf(fp, "addr=0x%llx, %s, size=%u, ", m_addr,
                m_write ? "store" : "load ", m_req_size);
        switch (m_type) {
        case GLOBAL_ACC_R:
            fprintf(fp, "GLOBAL_R");
            break;
        case LOCAL_ACC_R:
            fprintf(fp, "LOCAL_R ");
            break;
        case CONST_ACC_R:
            fprintf(fp, "CONST   ");
            break;
        case TEXTURE_ACC_R:
            fprintf(fp, "TEXTURE ");
            break;
        case GLOBAL_ACC_W:
            fprintf(fp, "GLOBAL_W");
            break;
        case LOCAL_ACC_W:
            fprintf(fp, "LOCAL_W ");
            break;
        case L2_WRBK_ACC:
            fprintf(fp, "L2_WRBK ");
            break;
        case INST_ACC_R:
            fprintf(fp, "INST    ");
            break;
        case L1_WRBK_ACC:
            fprintf(fp, "L1_WRBK ");
            break;
        default:
            fprintf(fp, "unknown ");
            break;
        }
    }

    gpgpu_context *gpgpu_ctx;

private:
    void init(gpgpu_context *ctx);

    /*该次访存操作的唯一 ID*/
    unsigned m_uid;

    /*访存地址*/
    new_addr_type m_addr;

    /*1-写，0-读*/
    bool m_write;

    /*访存数据大小，以byte为单位*/
    unsigned m_req_size;
    /**
      mem_access_type 定义了在时序模拟器中对不同类型的存储器进行不同的访存类型：
        MA_TUP(GLOBAL_ACC_R), 从 global memory 读
        MA_TUP(LOCAL_ACC_R), 从 local memory 读
        MA_TUP(CONST_ACC_R), 从常量缓存读
        MA_TUP(TEXTURE_ACC_R), 从纹理缓存读
        MA_TUP(GLOBAL_ACC_W), 向 global memory 写
        MA_TUP(LOCAL_ACC_W), 向 local memory 写
        MA_TUP(L2_WRBK_ACC), L2 缓存 write back
        MA_TUP(INST_ACC_R), 从指令缓存读
    */
    mem_access_type m_type;

    /*active masks of threads inside warp accessing the memory*/
    /*即因为 branch and memory divergences导致一个wrap中不是所有线程都有访存请求 */
    active_mask_t m_warp_mask;

    /**
     * mem_access_byte_mask_t 访存数据字节掩码定义：
     *  const unsigned MAX_MEMORY_ACCESS_SIZE = 128;
     *  typedef std::bitset<MAX_MEMORY_ACCESS_SIZE> mem_access_byte_mask_t;
     * 用于标记一次访存操作中的数据字节掩码，MAX_MEMORY_ACCESS_SIZE 设置为 128，
     * 即每次访存最大数据 128 字节，128 byte也是cache line的大小
     */
    mem_access_byte_mask_t m_byte_mask;

    /**
     * mem_access_sector_mask_t 扇区掩码定义：
        const unsigned SECTOR_CHUNCK_SIZE = 4;  // four sectors
        typedef std::bitset<SECTOR_CHUNCK_SIZE> mem_access_sector_mask_t;
        用于标记一次访存操作中的扇区掩码，4 个扇区，每个扇区 32 个字节数据。
     */
    mem_access_sector_mask_t m_sector_mask;
};

class mem_fetch;

class mem_fetch_interface {
public:
    virtual bool full(unsigned size, bool write) const = 0;
    virtual void push(mem_fetch *mf) = 0;
};

class mem_fetch_allocator {
public:
    virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                             unsigned size, bool wr, unsigned long long cycle,
                             unsigned long long streamID) const = 0;
    virtual mem_fetch *alloc(const class warp_inst_t &inst,
                             const mem_access_t &access,
                             unsigned long long cycle) const = 0;
    virtual mem_fetch *alloc(new_addr_type addr, mem_access_type type,
                             const active_mask_t &active_mask,
                             const mem_access_byte_mask_t &byte_mask,
                             const mem_access_sector_mask_t &sector_mask,
                             unsigned size, bool wr, unsigned long long cycle,
                             unsigned wid, unsigned sid, unsigned tpc,
                             mem_fetch *original_mf,
                             unsigned long long streamID) const = 0;
};

// the maximum number of destination, source, or address uarch operands in a
// instruction
#define MAX_REG_OPERANDS 32

struct dram_callback_t {
    dram_callback_t() {
        function = NULL;
        instruction = NULL;
        thread = NULL;
    }
    void (*function)(const class inst_t *, class ptx_thread_info *);

    const class inst_t *instruction;
    class ptx_thread_info *thread;
};

/**
 * 建模一条static
 * instruction和微架构相关的内容，主要包括静态信息，warp_inst_t则表达动态执行中的信息
 *  inst_t包含：
 *  - opcode type
 *  - source and destination register identifiers
 *  - instruction address
 *  - instruction size
 *  - reconvergence point instruction address
 *  - instruction latency and initiation interval
 *  - memory operations, the memory space accessed.
 */
class inst_t {
public:
    inst_t() {
        m_decoded = false;
        pc = (address_type)-1;
        reconvergence_pc = (address_type)-1;
        op = NO_OP;
        bar_type = NOT_BAR;
        red_type = NOT_RED;
        bar_id = (unsigned)-1;
        bar_count = (unsigned)-1;
        oprnd_type = UN_OP;
        sp_op = OTHER_OP;
        op_pipe = UNKOWN_OP;
        mem_op = NOT_TEX;
        const_cache_operand = 0;
        num_operands = 0;
        num_regs = 0;
        memset(out, 0, sizeof(unsigned));
        memset(in, 0, sizeof(unsigned));
        is_vectorin = 0;
        is_vectorout = 0;
        space = memory_space_t();
        cache_op = CACHE_UNDEFINED;
        latency = 1;
        initiation_interval = 1;
        for (unsigned i = 0; i < MAX_REG_OPERANDS; i++) {
            arch_reg.src[i] = -1;
            arch_reg.dst[i] = -1;
        }
        isize = 0;
    }
    bool valid() const { return m_decoded; }

    virtual void print_insn(FILE *fp) const {
        fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
    }

    /*指令的操作码是 LOAD 或 TENSOR_CORE_LOAD 则为load指令*/
    bool is_load() const {
        return (op == LOAD_OP || op == TENSOR_CORE_LOAD_OP ||
                memory_op == memory_load);
    }

    /*指令的操作码是 STORE 或 TENSOR_CORE_STORE 则为store指令*/
    bool is_store() const {
        return (op == STORE_OP || op == TENSOR_CORE_STORE_OP ||
                memory_op == memory_store);
    }

    bool is_fp() const { return ((sp_op == FP__OP)); } // VIJAY
    bool is_fpdiv() const { return ((sp_op == FP_DIV_OP)); }
    bool is_fpmul() const { return ((sp_op == FP_MUL_OP)); }
    bool is_dp() const { return ((sp_op == DP___OP)); }
    bool is_dpdiv() const { return ((sp_op == DP_DIV_OP)); }
    bool is_dpmul() const { return ((sp_op == DP_MUL_OP)); }
    bool is_imul() const { return ((sp_op == INT_MUL_OP)); }
    bool is_imul24() const { return ((sp_op == INT_MUL24_OP)); }
    bool is_imul32() const { return ((sp_op == INT_MUL32_OP)); }
    bool is_idiv() const { return ((sp_op == INT_DIV_OP)); }
    bool is_sfu() const {
        return ((sp_op == FP_SQRT_OP) || (sp_op == FP_LG_OP) ||
                (sp_op == FP_SIN_OP) || (sp_op == FP_EXP_OP) ||
                (sp_op == TENSOR__OP));
    }
    bool is_alu() const { return (sp_op == INT__OP); }

    unsigned get_num_operands() const { return num_operands; }
    unsigned get_num_regs() const { return num_regs; }
    void set_num_regs(unsigned num) { num_regs = num; }
    void set_num_operands(unsigned num) { num_operands = num; }
    void set_bar_id(unsigned id) { bar_id = id; }
    void set_bar_count(unsigned count) { bar_count = count; }

    /*program counter address of instruction */
    address_type pc;

    /*size of instruction in bytes */
    unsigned isize;

    /*opcode (uarch visible)*/
    op_type op;

    barrier_type bar_type;
    reduction_type red_type;
    unsigned bar_id;
    unsigned bar_count;

    /*code (uarch visible) identify if the operation is an interger or a floating point*/
    types_of_operands oprnd_type; 
    /*code (uarch visible) identify if int_alu, fp_alu, int_mul ....*/
    special_ops sp_op; 
    /*code (uarch visible) identify the pipeline of the operation (SP, SFU or MEM)*/
    operation_pipeline op_pipe; 
    /*code (uarch visible) identify memory type*/
    mem_operation mem_op;       
    /* has a load from constant memory as an operand*/
    bool const_cache_operand;  
    /*memory_op used by ptxplus*/
    _memory_op_t memory_op;    
    unsigned num_operands;
    /*count vector operand as one register operand*/
    unsigned num_regs; 

    /*-1 => not a branch, -2 => use function return address*/
    address_type reconvergence_pc; 

    /*目标寄存器ID*/
    unsigned out[8];
    unsigned outcount;

    /*源寄存器ID*/
    unsigned in[24];
    unsigned incount;
    unsigned char is_vectorin;
    unsigned char is_vectorout;
    int pred; // predicate register number
    int ar1, ar2;
    // register number for bank conflict evaluation
    struct {
        int dst[MAX_REG_OPERANDS];
        int src[MAX_REG_OPERANDS];
    } arch_reg;
    // int arch_reg[MAX_REG_OPERANDS]; // register number for bank conflict evaluation

    /*在在simd单元执行的latency相关*/
    unsigned latency; // operation latency
    unsigned initiation_interval;

    /*what is the size of the word being operated on*/
    unsigned data_size; 
    /*对于ld/st指令的访存特性*/
    memory_space_t space;
    cache_operator_type cache_op;

protected:
    bool m_decoded;
    virtual void pre_decode() {}
};

enum divergence_support_t { POST_DOMINATOR = 1, NUM_SIMD_MODEL };

/*MAX_ACCESSES_PER_INSN_PER_THREAD为单个线程中允许的最大访存次数。设置为8*/
const unsigned MAX_ACCESSES_PER_INSN_PER_THREAD = 8;

/**
 * warp_inst_t主要建模两个功能，主要用于时序仿真:
 *  1. 包含A dynamic “SIMD” instruction executed by a warp
    2. 用于issue()阶段后面的Pipeline Register
 * 继承自warp_inst_t 的每条ptx_instruction在 functional
 simulation中被填充，然后在时序模拟中ptx_instruction会被down成warp_inst_t从而丢掉不需要的信息
    包含以下内容：
        - warp_id
        - active thread mask inside the warp,
        - list of memory accesses (mem_access_t)
        - information of threads inside that warp (per_thread_info)
 */
class warp_inst_t : public inst_t {
public:
    // constructors
    warp_inst_t() {
        m_uid = 0;
        m_streamID = (unsigned long long)-1;
        m_empty = true;
        m_config = NULL;

        // Ni:
        m_is_ldgsts = false;
        m_is_ldgdepbar = false;
        m_is_depbar = false;

        m_depbar_group_no = 0;
    }
    warp_inst_t(const core_config *config) {
        m_uid = 0;
        m_streamID = (unsigned long long)-1;
        assert(config->warp_size <= MAX_WARP_SIZE);
        m_config = config;
        m_empty = true;
        m_isatomic = false;
        m_per_scalar_thread_valid = false;
        m_mem_accesses_created = false;
        m_cache_hit = false;
        m_is_printf = false;
        m_is_cdp = 0;
        should_do_atomic = true;

        // Ni:
        m_is_ldgsts = false;
        m_is_ldgdepbar = false;
        m_is_depbar = false;

        m_depbar_group_no = 0;
    }
    virtual ~warp_inst_t() {}

    // modifiers
    void broadcast_barrier_reduction(const active_mask_t &access_mask);
    void do_atomic(bool forceDo = false);
    void do_atomic(const active_mask_t &access_mask, bool forceDo = false);
    void clear() { m_empty = true; }

    /**
     * 设置指令动态发射过程中的一些信息:
     * mask：该指令对应的线程活跃掩码
     * warp_id：warp的id
     * cycle：发射的周期
     *  
     */
    void issue(const active_mask_t &mask, unsigned warp_id,
               unsigned long long cycle, int dynamic_warp_id, int sch_id,
               unsigned long long streamID);

    const active_mask_t &get_active_mask() const { return m_warp_active_mask; }
    void completed(unsigned long long cycle)
        const; // stat collection: called when the instruction is completed

    void set_addr(unsigned n, new_addr_type addr) {
        if (!m_per_scalar_thread_valid) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid = true;
        }
        m_per_scalar_thread[n].memreqaddr[0] = addr;
    }
    void set_addr(unsigned n, new_addr_type *addr, unsigned num_addrs) {
        if (!m_per_scalar_thread_valid) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid = true;
        }
        assert(num_addrs <= MAX_ACCESSES_PER_INSN_PER_THREAD);
        for (unsigned i = 0; i < num_addrs; i++)
            m_per_scalar_thread[n].memreqaddr[i] = addr[i];
    }
    void print_m_accessq() {
        if (accessq_empty())
            return;
        else {
            printf("Printing mem access generated\n");
            std::list<mem_access_t>::iterator it;
            for (it = m_accessq.begin(); it != m_accessq.end(); ++it) {
                printf("MEM_TXN_GEN:%s:%llx, Size:%d \n",
                       mem_access_type_str(it->get_type()), it->get_addr(),
                       it->get_size());
            }
        }
    }
    struct transaction_info {
        std::bitset<4> chunks; // bitmask: 32-byte chunks accessed
        mem_access_byte_mask_t bytes;
        active_mask_t active; // threads in this transaction

        bool test_bytes(unsigned start_bit, unsigned end_bit) {
            for (unsigned i = start_bit; i <= end_bit; i++)
                if (bytes.test(i))
                    return true;
            return false;
        }
    };

    void generate_mem_accesses();
    void memory_coalescing_arch(bool is_write, mem_access_type access_type);
    void memory_coalescing_arch_atomic(bool is_write,
                                       mem_access_type access_type);
    void memory_coalescing_arch_reduce_and_send(bool is_write,
                                                mem_access_type access_type,
                                                const transaction_info &info,
                                                new_addr_type addr,
                                                unsigned segment_size);

    void add_callback(unsigned lane_id,
                      void (*function)(const class inst_t *,
                                       class ptx_thread_info *),
                      const inst_t *inst, class ptx_thread_info *thread,
                      bool atomic) {
        if (!m_per_scalar_thread_valid) {
            m_per_scalar_thread.resize(m_config->warp_size);
            m_per_scalar_thread_valid = true;
            if (atomic)
                m_isatomic = true;
        }
        m_per_scalar_thread[lane_id].callback.function = function;
        m_per_scalar_thread[lane_id].callback.instruction = inst;
        m_per_scalar_thread[lane_id].callback.thread = thread;
    }
    void set_active(const active_mask_t &active);

    void clear_active(const active_mask_t &inactive);
    void set_not_active(unsigned lane_id);

    // accessors
    virtual void print_insn(FILE *fp) const {
        fprintf(fp, " [inst @ pc=0x%04llx] ", pc);
        for (int i = (int)m_config->warp_size - 1; i >= 0; i--)
            fprintf(fp, "%c", ((m_warp_active_mask[i]) ? '1' : '0'));
    }
    bool active(unsigned thread) const {
        return m_warp_active_mask.test(thread);
    }
    unsigned active_count() const { return m_warp_active_mask.count(); }
    unsigned issued_count() const {
        assert(m_empty == false);
        return m_warp_issued_mask.count();
    } // for instruction counting
    bool empty() const { return m_empty; }
    unsigned warp_id() const {
        assert(!m_empty);
        return m_warp_id;
    }
    unsigned warp_id_func() const // to be used in functional simulations only
    {
        return m_warp_id;
    }
    unsigned dynamic_warp_id() const {
        assert(!m_empty);
        return m_dynamic_warp_id;
    }
    bool has_callback(unsigned n) const {
        return m_warp_active_mask[n] && m_per_scalar_thread_valid &&
               (m_per_scalar_thread[n].callback.function != NULL);
    }
    new_addr_type get_addr(unsigned n) const {
        assert(m_per_scalar_thread_valid);
        return m_per_scalar_thread[n].memreqaddr[0];
    }

    bool isatomic() const { return m_isatomic; }

    unsigned warp_size() const { return m_config->warp_size; }

    bool accessq_empty() const { return m_accessq.empty(); }

    unsigned accessq_count() const { return m_accessq.size(); }

    const mem_access_t &accessq_back() { return m_accessq.back(); }
    
    void accessq_pop_back() { m_accessq.pop_back(); }

    /**
     * 一般的指令初始化时设置为 cycles = initiation_interval;，每次调用 dispatch_delay() 时，cycles 减 1，直到 cycles==0。
     * 但是有一个特例，就是当指令是访存 shared memory 时，为了模拟 bank conflict，需要将initiation_interval 设置为 total_accesses
     * 该函数模拟每周期访问一个bank
     */
    bool dispatch_delay() {
        if (cycles > 0)
            cycles--;
        return cycles > 0;
    }

    bool has_dispatch_delay() { return cycles > 0; }

    void print(FILE *fout) const;
    unsigned get_uid() const { return m_uid; }
    unsigned long long get_streamID() const { return m_streamID; }
    unsigned get_schd_id() const { return m_scheduler_id; }
    active_mask_t get_warp_active_mask() const { return m_warp_active_mask; }

protected:
    unsigned m_uid;
    unsigned long long m_streamID;
    /*指令为空*/
    bool m_empty;
    bool m_cache_hit;
    unsigned long long issue_cycle;

    /**用于实现initiation interval delay，即在m_dispatch_reg中等待的时间，或者访问shared memory的因为bank冲突而造成延迟 */
    unsigned cycles;
    bool m_isatomic;
    bool should_do_atomic;
    bool m_is_printf;
    unsigned m_warp_id;
    unsigned m_dynamic_warp_id;
    const core_config *m_config;

    /*dynamic active mask for timing model (after predication)*/
    active_mask_t m_warp_active_mask; 
    
    /*active mask at issue (prior to predication test) -- for instruction counting*/
    active_mask_t m_warp_issued_mask; 

    struct per_thread_info {
        per_thread_info() {
            for (unsigned i = 0; i < MAX_ACCESSES_PER_INSN_PER_THREAD; i++)
                memreqaddr[i] = 0;
        }
        dram_callback_t callback;
        /*每一个thread可以发出8个不同的request，support 32B access in 8 chunks of 4B each) */
        /*但是一般用memreqaddr[0]记录128 byte的访存地址 */
        new_addr_type memreqaddr[MAX_ACCESSES_PER_INSN_PER_THREAD]; 
    };

    bool m_per_scalar_thread_valid;
    
    /* information of threads inside that warp */
    std::vector<per_thread_info> m_per_scalar_thread;

    /*已经创建过 mem_access */
    bool m_mem_accesses_created;

    /*该指令对应的访问操作，即list of memory accesses*/
    std::list<mem_access_t> m_accessq;

    /**issue这条指令的scheduler id */
    unsigned m_scheduler_id;

    // Jin: cdp support
public:
    int m_is_cdp;

    // Ni: add boolean to indicate whether the instruction is ldgsts
    bool m_is_ldgsts;
    bool m_is_ldgdepbar;
    bool m_is_depbar;

    unsigned int m_depbar_group_no;
};

void move_warp(warp_inst_t *&dst, warp_inst_t *&src);

size_t get_kernel_code_size(class function_info *entry);
class checkpoint {
public:
    checkpoint();
    ~checkpoint() { printf("clasfsfss destructed\n"); }

    void load_global_mem(class memory_space *temp_mem, char *f1name);
    void store_global_mem(class memory_space *mem, char *fname, char *format);
    unsigned radnom;
};
/*
 * This abstract class used as a base for functional and performance and
 * simulation, it has basic functional simulation data structures and
 * procedures.
 */
class core_t {
public:
    core_t(gpgpu_sim *gpu, kernel_info_t *kernel, unsigned warp_size,
           unsigned threads_per_shader)
        : m_gpu(gpu), m_kernel(kernel), m_simt_stack(NULL), m_thread(NULL),
          m_warp_size(warp_size) {
        m_warp_count = threads_per_shader / m_warp_size;
        // Handle the case where the number of threads is not a
        // multiple of the warp size
        if (threads_per_shader % m_warp_size != 0) {
            m_warp_count += 1;
        }
        assert(m_warp_count * m_warp_size > 0);
        m_thread = (ptx_thread_info **)calloc(m_warp_count * m_warp_size,
                                              sizeof(ptx_thread_info *));
        initilizeSIMTStack(m_warp_count, m_warp_size);

        for (unsigned i = 0; i < MAX_CTA_PER_SHADER; i++) {
            for (unsigned j = 0; j < MAX_BARRIERS_PER_CTA; j++) {
                reduction_storage[i][j] = 0;
            }
        }
    }
    virtual ~core_t() { free(m_thread); }
    virtual void warp_exit(unsigned warp_id) = 0;
    virtual bool warp_waiting_at_barrier(unsigned warp_id) const = 0;
    virtual void checkExecutionStatusAndUpdate(warp_inst_t &inst, unsigned t,
                                               unsigned tid) = 0;
    class gpgpu_sim *get_gpu() { return m_gpu; }
    void execute_warp_inst_t(warp_inst_t &inst, unsigned warpId = (unsigned)-1);
    bool ptx_thread_done(unsigned hw_thread_id) const;
    virtual void updateSIMTStack(unsigned warpId, warp_inst_t *inst);
    void initilizeSIMTStack(unsigned warp_count, unsigned warps_size);
    void deleteSIMTStack();
    warp_inst_t getExecuteWarp(unsigned warpId);
    void get_pdom_stack_top_info(unsigned warpId, unsigned *pc,
                                 unsigned *rpc) const;
    kernel_info_t *get_kernel_info() { return m_kernel; }
    class ptx_thread_info **get_thread_info() { return m_thread; }
    unsigned get_warp_size() const { return m_warp_size; }
    void and_reduction(unsigned ctaid, unsigned barid, bool value) {
        reduction_storage[ctaid][barid] &= value;
    }
    void or_reduction(unsigned ctaid, unsigned barid, bool value) {
        reduction_storage[ctaid][barid] |= value;
    }
    void popc_reduction(unsigned ctaid, unsigned barid, bool value) {
        reduction_storage[ctaid][barid] += value;
    }
    unsigned get_reduction_value(unsigned ctaid, unsigned barid) {
        return reduction_storage[ctaid][barid];
    }

protected:
    /*时序模型*/
    class gpgpu_sim *m_gpu;

    /*运行在该simt core的kernel函数信息*/
    kernel_info_t *m_kernel;

    /*每一个warp对应一个simt堆栈*/
    simt_stack **m_simt_stack; // pdom based reconvergence context for each warp

    /*一维数组，用来标记core上执行的所有scala thread的信息，比如使用m_thread[tid]得到对应线程的信息*/
    class ptx_thread_info **m_thread;

    /*一个warp含有的线程数量*/
    unsigned m_warp_size;

    /*该simt core可以支持的warp数量，在core_t构造函数中设置为threads_per_shader / m_warp_size*/
    unsigned m_warp_count;
    
    unsigned reduction_storage[MAX_CTA_PER_SHADER][MAX_BARRIERS_PER_CTA];
};

/**
 * register that can hold multiple instructions.
 * 1个寄存器集合可以包含多条指令，方便模拟，而非真实硬件结构
 * 核心数据结构是：vector<warp_inst_t *> regs
 * 使用案例：
    其中在issue()阶段，The ID_OC Pipeline Register Set is used to hold the issued instructions 
    and waits them to be used by the operand collectors (OC).
 */
class register_set {
public:
    register_set(unsigned num, const char *name) {
        for (unsigned i = 0; i < num; i++) {
            regs.push_back(new warp_inst_t());
        }
        m_name = name;
    }
    const char *get_name() { return m_name; }
    bool has_free() {
        for (unsigned i = 0; i < regs.size(); i++) {
            if (regs[i]->empty()) {
                return true;
            }
        }
        return false;
    }
    bool has_free(bool sub_core_model, unsigned reg_id) {
        // in subcore model, each sched has a one specific reg to use (based on
        // sched id)
        if (!sub_core_model)
            return has_free();

        assert(reg_id < regs.size());
        return regs[reg_id]->empty();
    }
    bool has_ready() {
        for (unsigned i = 0; i < regs.size(); i++) {
            if (not regs[i]->empty()) {
                return true;
            }
        }
        return false;
    }
    bool has_ready(bool sub_core_model, unsigned reg_id) {
        if (!sub_core_model)
            return has_ready();
        assert(reg_id < regs.size());
        return (not regs[reg_id]->empty());
    }

    unsigned get_ready_reg_id() {
        // for sub core model we need to figure which reg_id has the ready warp
        // this function should only be called if has_ready() was true
        assert(has_ready());
        warp_inst_t **ready;
        ready = NULL;
        unsigned reg_id = 0;
        for (unsigned i = 0; i < regs.size(); i++) {
            if (not regs[i]->empty()) {
                if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
                    // ready is oldest
                } else {
                    ready = &regs[i];
                    reg_id = i;
                }
            }
        }
        return reg_id;
    }
    unsigned get_schd_id(unsigned reg_id) {
        assert(not regs[reg_id]->empty());
        return regs[reg_id]->get_schd_id();
    }

    /*找到寄存器组中空闲的register，将src中内容的移入空闲的寄存器 */
    void move_in(warp_inst_t *&src) {
        warp_inst_t **free = get_free();
        move_warp(*free, src);
    }
    // void copy_in( warp_inst_t* src ){
    //   src->copy_contents_to(*get_free());
    //}
    void move_in(bool sub_core_model, unsigned reg_id, warp_inst_t *&src) {
        warp_inst_t **free;
        if (!sub_core_model) {
            free = get_free();
        } else {
            assert(reg_id < regs.size());
            free = get_free(sub_core_model, reg_id);
        }
        move_warp(*free, src);
    }

    void move_out_to(warp_inst_t *&dest) {
        warp_inst_t **ready = get_ready();
        move_warp(dest, *ready);
    }
    void move_out_to(bool sub_core_model, unsigned reg_id, warp_inst_t *&dest) {
        if (!sub_core_model) {
            return move_out_to(dest);
        }
        warp_inst_t **ready = get_ready(sub_core_model, reg_id);
        assert(ready != NULL);
        move_warp(dest, *ready);
    }

    warp_inst_t **get_ready() {
        warp_inst_t **ready;
        ready = NULL;
        for (unsigned i = 0; i < regs.size(); i++) {
            if (not regs[i]->empty()) {
                if (ready and (*ready)->get_uid() < regs[i]->get_uid()) {
                    // ready is oldest
                } else {
                    ready = &regs[i];
                }
            }
        }
        return ready;
    }
    warp_inst_t **get_ready(bool sub_core_model, unsigned reg_id) {
        if (!sub_core_model)
            return get_ready();
        warp_inst_t **ready;
        ready = NULL;
        assert(reg_id < regs.size());
        if (not regs[reg_id]->empty())
            ready = &regs[reg_id];
        return ready;
    }

    void print(FILE *fp) const {
        fprintf(fp, "%s : @%p\n", m_name, this);
        for (unsigned i = 0; i < regs.size(); i++) {
            fprintf(fp, "     ");
            regs[i]->print(fp);
            fprintf(fp, "\n");
        }
    }

    /*找到一个空的register */
    warp_inst_t **get_free() {
        for (unsigned i = 0; i < regs.size(); i++) {
            if (regs[i]->empty()) {
                return &regs[i];
            }
        }
        assert(0 && "No free registers found");
        return NULL;
    }

    /**
     * 找到一个空闲的register，对于subcore model，则采用reg_id指定的register
     */
    warp_inst_t **get_free(bool sub_core_model, unsigned reg_id) {
        // in subcore model, each sched has a one specific reg to use (based on sched id)
        if (!sub_core_model)
            return get_free();

        assert(reg_id < regs.size());
        if (regs[reg_id]->empty()) {
            return &regs[reg_id];
        }
        assert(0 && "No free register found");
        return NULL;
    }

    unsigned get_size() { return regs.size(); }

private:
    /*寄存器合集，每一个register保存warp_inst_t指令*/
    std::vector<warp_inst_t *> regs;

    /*register set名称 */
    const char *m_name;
};

#endif // #ifdef __cplusplus

#endif // #ifndef ABSTRACT_HARDWARE_MODEL_INCLUDED
