# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Streaming DRAM bandwidth probe for the B390: how fast can we *read* ~400 MiB with a
perfectly coalesced, aligned kernel, flushing L2 between iters (same method as bench.py)?

This establishes the achievable-BW ceiling the dp4a decode GEMV is competing against.
"""
import os, sys, statistics
import numpy as np
import pyopencl as cl

MIB = int(sys.argv[1]) if len(sys.argv) > 1 else 398   # ~ weight size (248320*8*210 = 397.8 MiB)
ITERS, WARMUP = 50, 40
FLUSH_MIB = 128

SRC = r"""
__attribute__((intel_reqd_sub_group_size(16)))
__kernel void stream_read(__global const uint16* src, uint n16, __global uint* sink) {
    uint gid = get_global_id(0);
    uint gsz = get_global_size(0);
    uint16 acc = (uint16)(0);
    for (uint i = gid; i < n16; i += gsz)
        acc += src[i];                     // coalesced 64-byte aligned loads, all lanes active
    uint s = acc.s0+acc.s1+acc.s2+acc.s3+acc.s4+acc.s5+acc.s6+acc.s7
           + acc.s8+acc.s9+acc.sa+acc.sb+acc.sc+acc.sd+acc.se+acc.sf;
    if (gid == 0) sink[0] = s;
}
"""
FLUSH_SRC = r"""
__kernel void cache_flush(__global float* buf, uint n, __global float* sink) {
    uint gid = get_global_id(0), gsz = get_global_size(0); float acc = 0.0f;
    for (uint i = gid; i < n; i += gsz) { buf[i] += 1.0f; acc += buf[i]; }
    if (gid == 0) sink[0] = acc;
}
"""

plat = cl.get_platforms()[0]
dev = plat.get_devices()[0]
ctx = cl.Context([dev])
q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
print("Device:", dev.name)

nbytes = MIB * 1024 * 1024
n16 = nbytes // 64
mf = cl.mem_flags
src = cl.Buffer(ctx, mf.READ_ONLY, size=n16 * 64)
sink = cl.Buffer(ctx, mf.WRITE_ONLY, size=64)
cl.enqueue_copy(q, src, np.ones(n16 * 16, np.uint32))
prog = cl.Program(ctx, SRC).build()
k = cl.Kernel(prog, "stream_read")

# flush setup
fe = (FLUSH_MIB * 1024 * 1024) // 4
fb = cl.Buffer(ctx, mf.READ_WRITE, size=fe * 4)
fs = cl.Buffer(ctx, mf.WRITE_ONLY, size=4096 * 4)
cl.enqueue_copy(q, fb, np.zeros(fe, np.float32))
fprog = cl.Program(ctx, FLUSH_SRC).build()
fk = cl.Kernel(fprog, "cache_flush")
q.finish()

def flush():
    for _ in range(2):
        fk(q, (min(fe, 1024 * 1024),), None, fb, np.uint32(fe), fs)
    q.finish()

gsz = (1024 * 256,)   # plenty of work-items
def enq():
    return k(q, gsz, (16,), src, np.uint32(n16), sink)

for _ in range(WARMUP):
    enq(); q.finish()
times = []
for _ in range(ITERS):
    flush()
    ev = enq(); ev.wait()
    times.append((ev.profile.end - ev.profile.start) * 1e-9)
med = statistics.median(times)
print("read %d MiB  median=%.4f ms  BW=%.1f GB/s" % (MIB, med * 1e3, nbytes / med / 1e9))


def tiled_probe():
    """Mimic the dp4a kernel's access: many subgroups, each stages a ROW_BYTES tile into SLM
    (coalesced uint4) with a barrier, then 'consumes' it (sum) — no decode. Isolates the BW
    achievable by the staged-tile structure vs the flat streaming probe."""
    ROW = 1680                      # bytes per row (8 blocks * 210)
    NROWT = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    Nrows = (MIB * 1024 * 1024) // (ROW * NROWT) * NROWT   # total rows (multiple of NROWT)
    total_bytes = Nrows * ROW
    src2 = cl.Buffer(ctx, mf.READ_ONLY, size=total_bytes)
    cl.enqueue_copy(q, src2, np.ones(total_bytes // 4, np.uint32)); q.finish()
    tsrc = r"""
    #define ROW 1680
    #define NROWT %d
    __attribute__((intel_reqd_sub_group_size(16)))
    __kernel void tiled(__global const uint* W, uint nrows, __global uint* sink) {
        __local uchar buf[NROWT*ROW];
        int lane = get_sub_group_local_id();
        int sg = get_global_id(0) / 16;
        uint n0 = (uint)sg * NROWT;
        if (n0 >= nrows) return;
        __local uint* bu = (__local uint*)buf;
        const __global uint* ws = W + (uint)n0 * (ROW/4);
        uint tu4 = (uint)NROWT * ROW / 16u;
        for (uint i = lane; i < tu4; i += 16) vstore4(vload4(i, ws), i, bu);
        barrier(CLK_LOCAL_MEM_FENCE);
        uint acc = 0;
        for (uint i = lane; i < NROWT*ROW/4; i += 16) acc += bu[i];
        if (lane == 0 && sg == 0) sink[0] = acc;
    }
    """ % NROWT
    tp = cl.Program(ctx, tsrc).build()
    tk = cl.Kernel(tp, "tiled")
    nsg = Nrows // NROWT
    gsz2 = (nsg * 16,)
    def enq2():
        return tk(q, gsz2, (16,), src2, np.uint32(Nrows), sink)
    for _ in range(WARMUP): enq2(); q.finish()
    ts = []
    for _ in range(ITERS):
        flush(); ev = enq2(); ev.wait(); ts.append((ev.profile.end-ev.profile.start)*1e-9)
    med = statistics.median(ts)
    print("tiled NROWT=%d  read %d MiB  median=%.4f ms  BW=%.1f GB/s"
          % (NROWT, total_bytes//1048576, med*1e3, total_bytes/med/1e9))

if len(sys.argv) > 2 and sys.argv[2] == "tiled":
    tiled_probe()


def tiled16_probe():
    """Tiled staging like tiled_probe but with 64-byte uint16 coalesced loads (NROWT=4 -> tile base
    is 64-aligned). Tests whether wider loads raise the staged-tile BW ceiling."""
    ROW = 1680; NROWT = 4
    Nrows = (MIB*1024*1024)//(ROW*NROWT)*NROWT
    total_bytes = Nrows*ROW
    src2 = cl.Buffer(ctx, mf.READ_ONLY, size=total_bytes)
    cl.enqueue_copy(q, src2, np.ones(total_bytes//4, np.uint32)); q.finish()
    tsrc = r"""
    #define ROW 1680
    #define NROWT 4
    __attribute__((intel_reqd_sub_group_size(16)))
    __kernel void tiled16(__global const uint* W, uint nrows, __global uint* sink){
        __local uchar buf[NROWT*ROW];
        int lane=get_sub_group_local_id(); int sg=get_global_id(0)/16;
        uint n0=(uint)sg*NROWT; if(n0>=nrows) return;
        __local uint* bu=(__local uint*)buf;
        const __global uint* ws=W+(uint)n0*(ROW/4);
        uint tu16=(uint)NROWT*ROW/64u;                 // # of 64B chunks
        for(uint i=lane;i<tu16;i+=16) vstore16(vload16(i,ws),i,bu);
        barrier(CLK_LOCAL_MEM_FENCE);
        uint acc=0; for(uint i=lane;i<NROWT*ROW/4;i+=16) acc+=bu[i];
        if(lane==0&&sg==0) sink[0]=acc;
    }
    """
    tp=cl.Program(ctx,tsrc).build(); tk=cl.Kernel(tp,"tiled16")
    nsg=Nrows//NROWT; gsz2=(nsg*16,)
    def enq2(): return tk(q,gsz2,(16,),src2,np.uint32(Nrows),sink)
    for _ in range(WARMUP): enq2(); q.finish()
    ts=[]
    for _ in range(ITERS):
        flush(); ev=enq2(); ev.wait(); ts.append((ev.profile.end-ev.profile.start)*1e-9)
    med=statistics.median(ts)
    print("tiled16 NROWT=4 read %d MiB median=%.4f ms BW=%.1f GB/s"%(total_bytes//1048576,med*1e3,total_bytes/med/1e9))

if len(sys.argv)>2 and sys.argv[2]=="tiled16":
    tiled16_probe()
