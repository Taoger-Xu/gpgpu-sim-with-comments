// Copyright (c) 2009-2011, Tor M. Aamodt, Inderpreet Singh
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

#include "assert.h"
#include <set>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#ifndef SCOREBOARD_H_
#define SCOREBOARD_H_

#include "../abstract_hardware_model.h"

/**
 * 每一个wrap对应一个scoreboard，实现一个wrap内的顺序发射，乱序执行
 * 在issue()的时候观察该instruction要写入的寄存器从而修改scoreboard
 */
class Scoreboard {
public:
    Scoreboard(unsigned sid, unsigned n_warps, class gpgpu_t *gpu);

    void reserveRegisters(const warp_inst_t *inst);
    void releaseRegisters(const warp_inst_t *inst);
    void releaseRegister(unsigned wid, unsigned regnum);

    bool checkCollision(unsigned wid, const inst_t *inst) const;
    bool pendingWrites(unsigned wid) const;
    void printContents() const;
    const bool islongop(unsigned warp_id, unsigned regnum);

private:
    void reserveRegister(unsigned wid, unsigned regnum);
    int get_sid() const { return m_sid; }

    /*simt core id */
    unsigned m_sid;

    /**
     * reg_table reserves all the destination registers in the issued instructions that are not written back
     * keeps track of pending writes to registers
     * tracks all the destination registers
     * indexed by warp id, reg_id => pending write count
     */
    std::vector<std::set<unsigned>> reg_table;

    // Register that depend on a long operation (global, local or tex memory)
    /*longopregs reserves all the destination registers in the issued memory access instructions that are not written back.*/
    std::vector<std::set<unsigned>> longopregs;

    class gpgpu_t *m_gpu;
};

#endif /* SCOREBOARD_H_ */
