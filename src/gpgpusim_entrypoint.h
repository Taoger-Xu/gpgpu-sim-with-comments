// Copyright (c) 2009-2011, Tor M. Aamodt
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

#ifndef GPGPUSIM_ENTRYPOINT_H_INCLUDED
#define GPGPUSIM_ENTRYPOINT_H_INCLUDED

#include "abstract_hardware_model.h"
#include <pthread.h>
#include <semaphore.h>
#include <time.h>

// extern time_t g_simulation_starttime;
class gpgpu_context;

/**
 * GPGPUsim_ctx是performance simulation model的进一步封装,主要控制如何进行时序模拟：
 * 比如时序模拟的开始，以及需要模拟的operation
 * 即通过g_stream_manager不断读取CUDA API执行的op,然后通过g_the_gpu进行时序模拟
 * 包含全局唯一的g_the_gpu和g_stream_manager,
 * 同时包含启动和结束模拟的各种信号量
 */
class GPGPUsim_ctx {
public:
    /**初始化 设置g_sim_done为true */
    GPGPUsim_ctx(gpgpu_context *ctx) {
        g_sim_active = false;
        g_sim_done = true;
        break_limit = false;
        g_sim_lock = PTHREAD_MUTEX_INITIALIZER;

        g_the_gpu_config = NULL;
        g_the_gpu = NULL;
        g_stream_manager = NULL;
        the_cude_device = NULL;
        the_context = NULL;
        gpgpu_ctx = ctx;
    }

    // struct gpgpu_ptx_sim_arg *grid_params;

    sem_t g_sim_signal_start;
    sem_t g_sim_signal_finish;
    sem_t g_sim_signal_exit;
    time_t g_simulation_starttime;
    pthread_t g_simulation_thread;

    /*全局唯一的配置文件信息 */
    class gpgpu_sim_config *g_the_gpu_config;

    /*全局唯一的gpgpu_sim */
    class gpgpu_sim *g_the_gpu;

    /*全局唯一的stream_manager，在cuda_runtime_api.cc中实现的各种API中压入新的op，然后去执行时序模拟 */
    class stream_manager *g_stream_manager;

    /*该gpgpu对应的device id */
    struct _cuda_device_id *the_cude_device;

    /*全局只有一个CUctx_st*/
    struct CUctx_st *the_context;
    
    gpgpu_context *gpgpu_ctx;

    pthread_mutex_t g_sim_lock;
    bool g_sim_active;

    /**gpgpu-sim时序模拟结束的信号，在GPGPUsim_ctx()构造函数中设置为true */
    bool g_sim_done;
    bool break_limit;
};

#endif
