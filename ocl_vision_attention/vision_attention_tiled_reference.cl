// [REFERENCE_START]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define REDUCE_LANES 32
#define KEY_TILE 16
#define MAX_HEAD_DIM 512

__kernel void vision_attention_tiled_f16(
    __global const half* Q,
    __global const half* K,
    __global const half* V,
    __global half* output,
    const ulong q_offset,
    const ulong k_offset,
    const ulong v_offset,
    const ulong out_offset,
    const uint seq_len,
    const uint num_heads,
    const uint head_dim,
    const uint attention_window,
    const float scale) {
  const uint lane = get_local_id(0);
  const uint key_lane = get_local_id(1);
  const uint head = get_group_id(1);
  const uint query_row = get_group_id(2);
  const uint local_flat = key_lane * REDUCE_LANES + lane;

  if (lane >= REDUCE_LANES || key_lane >= KEY_TILE ||
      head >= num_heads || query_row >= seq_len) {
    return;
  }

  __local float q_cache[MAX_HEAD_DIM];
  __local float dot_partials[KEY_TILE * REDUCE_LANES];
  __local float scores[KEY_TILE];
  __local float weights[KEY_TILE];
  __local float shared_max;
  __local float shared_denom;
  __local float shared_alpha;

  if (local_flat < head_dim) {
    const ulong q_index = (((ulong)query_row * num_heads + head) * head_dim) + local_flat;
    q_cache[local_flat] = convert_float(Q[q_offset + q_index]);
  }
  if (local_flat == 0) {
    shared_max = -3.402823466e+38f;
    shared_denom = 0.0f;
    shared_alpha = 0.0f;
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  uint kv_start = 0;
  uint kv_end = seq_len;
  if (attention_window != 0) {
    kv_start = (query_row / attention_window) * attention_window;
    kv_end = min(kv_start + attention_window, seq_len);
  }

  float acc = 0.0f;
  const uint out_dim = local_flat;

  for (uint tile_start = kv_start; tile_start < kv_end; tile_start += KEY_TILE) {
    const uint kv_row = tile_start + key_lane;
    float partial = 0.0f;
    if (kv_row < kv_end) {
      for (uint d = lane; d < head_dim; d += REDUCE_LANES) {
        const ulong k_index = (((ulong)kv_row * num_heads + head) * head_dim) + d;
        partial += q_cache[d] * convert_float(K[k_offset + k_index]);
      }
    }
    dot_partials[key_lane * REDUCE_LANES + lane] = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (uint stride = REDUCE_LANES >> 1; stride > 0; stride >>= 1) {
      if (lane < stride) {
        const uint idx = key_lane * REDUCE_LANES + lane;
        dot_partials[idx] += dot_partials[idx + stride];
      }
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lane == 0) {
      scores[key_lane] = (kv_row < kv_end)
          ? dot_partials[key_lane * REDUCE_LANES] * scale
          : -3.402823466e+38f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_flat == 0) {
      float tile_max = -3.402823466e+38f;
      for (uint i = 0; i < KEY_TILE; ++i) {
        tile_max = fmax(tile_max, scores[i]);
      }
      const float new_max = fmax(shared_max, tile_max);
      const float alpha = (shared_denom == 0.0f) ? 0.0f : exp(shared_max - new_max);
      float tile_denom = 0.0f;
      for (uint i = 0; i < KEY_TILE; ++i) {
        const float beta = exp(scores[i] - new_max);
        weights[i] = beta;
        tile_denom += beta;
      }
      shared_alpha = alpha;
      shared_max = new_max;
      shared_denom = shared_denom * alpha + tile_denom;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (out_dim < head_dim) {
      float tile_acc = 0.0f;
      for (uint i = 0; i < KEY_TILE; ++i) {
        const uint value_row = tile_start + i;
        if (value_row < kv_end) {
          const ulong v_index = (((ulong)value_row * num_heads + head) * head_dim) + out_dim;
          tile_acc += weights[i] * convert_float(V[v_offset + v_index]);
        }
      }
      acc = acc * shared_alpha + tile_acc;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (out_dim < head_dim) {
    const ulong out_idx = (((ulong)query_row * num_heads + head) * head_dim) + out_dim;
    output[out_offset + out_idx] = convert_half_rte(acc / shared_denom);
  }
}

// [REFERENCE_END]
