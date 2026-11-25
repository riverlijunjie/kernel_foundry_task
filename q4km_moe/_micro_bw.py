import numpy as np, pyopencl as cl, statistics, os

SRC = r"""
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#define OPG 16
#ifndef KSPLIT
#define KSPLIT 8
#endif

__attribute__((intel_reqd_sub_group_size(OPG)))
__kernel void memread(
    __global const uint * restrict ups,
    __global const uint * restrict gates,
    __global const uint * restrict indexes,
    __global       float * restrict outputs,
    uint output_num, uint nbpr, uint num_blocks)
{
    const uint h  = get_group_id(0);
    const uint hh = get_local_id(0);
    const uint sg = get_local_id(1);
    const uint v  = get_group_id(2) % output_num;
    const uint index = indexes[v];
    const uint expert_u = num_blocks * 144 / 4;
    const __global uint *up = ups   + (size_t)index * expert_u;
    const __global uint *gt = gates + (size_t)index * expert_u;
    const __global uint *up_sl = up + num_blocks * 32;
    const __global uint *gt_sl = gt + num_blocks * 32;
    uint acc = 0;
    for (uint bid = sg; bid < nbpr; bid += KSPLIT) {
        uint pe = (h * nbpr + bid) * (OPG * 128 / 4);
        uint po = (h * nbpr + bid) * (OPG * 16 / 4);
        uint8 a0 = intel_sub_group_block_read8(up + pe +   0);
        uint8 a1 = intel_sub_group_block_read8(up + pe + 128);
        uint8 a2 = intel_sub_group_block_read8(up + pe + 256);
        uint8 a3 = intel_sub_group_block_read8(up + pe + 384);
        uint4 a4 = intel_sub_group_block_read4(up_sl + po);
        uint8 b0 = intel_sub_group_block_read8(gt + pe +   0);
        uint8 b1 = intel_sub_group_block_read8(gt + pe + 128);
        uint8 b2 = intel_sub_group_block_read8(gt + pe + 256);
        uint8 b3 = intel_sub_group_block_read8(gt + pe + 384);
        uint4 b4 = intel_sub_group_block_read4(gt_sl + po);
        acc += a0.s0+a0.s1+a0.s2+a0.s3+a0.s4+a0.s5+a0.s6+a0.s7;
        acc += a1.s0+a1.s1+a1.s2+a1.s3+a1.s4+a1.s5+a1.s6+a1.s7;
        acc += a2.s0+a2.s1+a2.s2+a2.s3+a2.s4+a2.s5+a2.s6+a2.s7;
        acc += a3.s0+a3.s1+a3.s2+a3.s3+a3.s4+a3.s5+a3.s6+a3.s7;
        acc += a4.s0+a4.s1+a4.s2+a4.s3;
        acc += b0.s0+b0.s1+b0.s2+b0.s3+b0.s4+b0.s5+b0.s6+b0.s7;
        acc += b1.s0+b1.s1+b1.s2+b1.s3+b1.s4+b1.s5+b1.s6+b1.s7;
        acc += b2.s0+b2.s1+b2.s2+b2.s3+b2.s4+b2.s5+b2.s6+b2.s7;
        acc += b3.s0+b3.s1+b3.s2+b3.s3+b3.s4+b3.s5+b3.s6+b3.s7;
        acc += b4.s0+b4.s1+b4.s2+b4.s3;
    }
    if (acc == 0xdeadbeefu) outputs[h*OPG+hh] = 1.0f;
}

__attribute__((intel_reqd_sub_group_size(16)))
__kernel void stream(__global const uint* restrict src, uint n_u8blocks,
                     __global float* restrict out) {
    uint gid = get_global_id(0) / 16;
    uint nsg = get_global_size(0) / 16;
    uint acc = 0;
    for (uint b = gid; b < n_u8blocks; b += nsg) {
        uint8 v = intel_sub_group_block_read8(src + b * 128);
        acc += v.s0+v.s1+v.s2+v.s3+v.s4+v.s5+v.s6+v.s7;
    }
    if (acc == 0xdeadbeefu) out[0] = 1.0f;
}

__kernel void cache_flush(__global float* buf, uint n, __global float* sink) {
    uint gid = get_global_id(0); uint stride = get_global_size(0);
    float acc = 0.0f;
    for (uint i = gid; i < n; i += stride) acc += buf[i];
    if (gid < 4096u) sink[gid] += acc;
}
"""

dev = cl.get_platforms()[0].get_devices()[0]
ctx = cl.Context([dev])
q = cl.CommandQueue(ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)
KS = int(os.environ.get("KSPLIT", 8))
prog = cl.Program(ctx, SRC).build(options="-cl-std=CL2.0 -DKSPLIT=%d" % KS)
mf = cl.mem_flags

K, N, ne, active = 2048, 512, 256, 8
nbpr = K // 256
num_blocks = K * N // 256
expert_size = num_blocks * 144
pool = np.random.randint(0, 255, ne * expert_size, dtype=np.uint8)
ups = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pool)
gts = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=pool)
idx = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,
                hostbuf=np.random.choice(ne, active, replace=False).astype(np.uint32))
out = cl.Buffer(ctx, mf.WRITE_ONLY, size=4 * N)

fl_n = 128 * 1024 * 1024 // 4
flbuf = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.zeros(fl_n, np.float32))
flsink = cl.Buffer(ctx, mf.WRITE_ONLY, size=4096 * 4)
kfl = cl.Kernel(prog, "cache_flush")


def flush():
    for _ in range(2):
        kfl(q, (1 << 20,), None, flbuf, np.uint32(fl_n), flsink)
    q.finish()


def bench(enq, iters=100):
    for _ in range(10):
        enq()
        q.finish()
    ts = []
    for _ in range(iters):
        flush()
        ev = enq()
        ev.wait()
        ts.append((ev.profile.end - ev.profile.start) * 1e-9)
    return statistics.median(ts)


k = cl.Kernel(prog, "memread")
k.set_args(ups, gts, idx, out, np.uint32(active), np.uint32(nbpr), np.uint32(num_blocks))
gs, ls = (N, KS, active), (16, KS, 1)
t = bench(lambda: cl.enqueue_nd_range_kernel(q, k, gs, ls))
moved = active * 2 * expert_size
print("memread  KSPLIT=%d: %.4f ms   %.1f GB/s  (moved %.1f MB)" %
      (KS, t * 1e3, moved / t / 1e9, moved / 1e6))

ks = cl.Kernel(prog, "stream")
nb = active * 2 * expert_size // 512
for nsg in (256, 512, 1024, 2048, 4096, 8192):
    ks.set_args(ups, np.uint32(nb), out)
    t2 = bench(lambda: cl.enqueue_nd_range_kernel(q, ks, (nsg * 16,), (256,)), iters=50)
    print("stream %d blocks (%.1f MB) nsg=%6d: %.4f ms  %.1f GB/s" %
          (nb, nb * 512 / 1e6, nsg, t2 * 1e3, nb * 512 / t2 / 1e9))
