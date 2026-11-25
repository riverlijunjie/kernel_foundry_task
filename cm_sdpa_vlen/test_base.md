(venv) (base) intel@intel-Xiaomi-Book-Pro-14:~/river/aboutSHW/opencl/tests/pageatten$ PA_OMNI=1 python3 test_pa.py
[100%] Built target csrc
 GPU device [0] : Intel(R) Graphics by "Intel(R) Corporation" : 32 EUs 2450 MHz 131072 bytes SLM [DPAS][sbarrier]
CM_GRF_WIDTH=np.int32(512)
compiling /home/intel/river/aboutSHW/opencl/tests/pageatten num_heads=32 head_size=128 sparse_block_sz=1 self.compressed_kvcache=0...
[total]: PA_causal sparse_block_sz=1 seq_len=902 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 4,131 GFLOPS, latency(ms): min=0.877 avg=1.613 max=2.974, 20.7% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=894 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 8,375 GFLOPS, latency(ms): min=0.771 avg=0.782 max=0.807, 41.9% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=1534 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 11,931 GFLOPS, latency(ms): min=1.597 avg=1.616 max=1.642, 59.7% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2226 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 11,260 GFLOPS, latency(ms): min=3.586 avg=3.605 max=3.628, 56.3% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=14910 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 15,166 GFLOPS, latency(ms): min=119.030 avg=120.083 max=121.102, 75.8% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2556 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 0, MFU 13,243 GFLOPS, latency(ms): min=4.005 avg=4.041 max=4.067, 66.2% of 20 TFLOPS XMX peak
compiling /home/intel/river/aboutSHW/opencl/tests/pageatten num_heads=32 head_size=128 sparse_block_sz=1 self.compressed_kvcache=1...
[total]: PA_causal sparse_block_sz=1 seq_len=902 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 5,032 GFLOPS, latency(ms): min=1.300 avg=1.324 max=1.467, 25.2% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=894 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 6,224 GFLOPS, latency(ms): min=1.045 avg=1.052 max=1.073, 31.1% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=1534 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 8,428 GFLOPS, latency(ms): min=2.275 avg=2.287 max=2.309, 42.1% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2226 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 8,343 GFLOPS, latency(ms): min=4.829 avg=4.865 max=4.921, 41.7% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=14910 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 10,109 GFLOPS, latency(ms): min=179.864 avg=180.157 max=180.473, 50.5% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2556 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 1, MFU 9,007 GFLOPS, latency(ms): min=5.910 avg=5.942 max=6.013, 45.0% of 20 TFLOPS XMX peak
compiling /home/intel/river/aboutSHW/opencl/tests/pageatten num_heads=32 head_size=128 sparse_block_sz=1 self.compressed_kvcache=2...
[total]: PA_causal sparse_block_sz=1 seq_len=902 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 4,831 GFLOPS, latency(ms): min=1.084 avg=1.380 max=1.806, 24.2% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=894 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 6,244 GFLOPS, latency(ms): min=1.031 avg=1.049 max=1.098, 31.2% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=1534 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 8,264 GFLOPS, latency(ms): min=2.249 avg=2.333 max=2.742, 41.3% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2226 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 8,184 GFLOPS, latency(ms): min=4.853 avg=4.960 max=5.381, 40.9% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=14910 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 9,922 GFLOPS, latency(ms): min=181.216 avg=183.554 max=189.302, 49.6% of 20 TFLOPS XMX peak
[total]: PA_causal sparse_block_sz=1 seq_len=2556 , num_trunks=1, density req/eff 1.00/1.00, compressKVCache 2, MFU 9,235 GFLOPS, latency(ms): min=5.738 avg=5.795 max=5.842, 46.2% of 20 TFLOPS XMX peak

==========================================================================================
  Qwen3-Omni-4B Thinker LM Prefill — PA Kernel HW Utilization on PTL 4Xe
  32Q/8KV GQA, HD=128, KV cache: FP16 (NONE)  |  Peak: 20.00 TFLOPS
==========================================================================================
  Case                          S (tokens)   PA avg (ms)   Efficiency   PA / ref (ms)
------------------------------------------------------------------------------------------
  C1: 448×448×2                        902          1.61        20.7%        11.997 ms @100%
  C2: 512×384×2                        894          0.78        41.9%        11.785 ms @100%
  C3: 1024×512×2                      1534          1.62        59.7%        34.699 ms @100%
  C4: 1260×700×2                      2226          3.61        56.3%        73.066 ms @100%
  C5: 1280×768×15                    14910        120.08        75.8%      3278.066 ms @100%
  C6: 1260×700×2 mt512                2556          4.04        66.2%        96.335 ms @100%
==========================================================================================

==========================================================================================
  Qwen3-Omni-4B Thinker LM Prefill — PA Kernel HW Utilization on PTL 4Xe
  32Q/8KV GQA, HD=128, KV cache: INT8 (BY_TOKEN)  |  Peak: 20.00 TFLOPS
==========================================================================================
  Case                          S (tokens)   PA avg (ms)   Efficiency   PA / ref (ms)
------------------------------------------------------------------------------------------
  C1: 448×448×2                        902          1.32        25.2%        11.997 ms @100%
  C2: 512×384×2                        894          1.05        31.1%        11.785 ms @100%
  C3: 1024×512×2                      1534          2.29        42.1%        34.699 ms @100%
  C4: 1260×700×2                      2226          4.87        41.7%        73.066 ms @100%
  C5: 1280×768×15                    14910        180.16        50.5%      3278.066 ms @100%
  C6: 1260×700×2 mt512                2556          5.94        45.0%        96.335 ms @100%
==========================================================================================

==========================================================================================
  Qwen3-Omni-4B Thinker LM Prefill — PA Kernel HW Utilization on PTL 4Xe
  32Q/8KV GQA, HD=128, KV cache: INT8 (BY_CHANNEL)  |  Peak: 20.00 TFLOPS
==========================================================================================
  Case                          S (tokens)   PA avg (ms)   Efficiency   PA / ref (ms)
------------------------------------------------------------------------------------------
  C1: 448×448×2                        902          1.38        24.2%        11.997 ms @100%
  C2: 512×384×2                        894          1.05        31.2%        11.785 ms @100%
  C3: 1024×512×2                      1534          2.33        41.3%        34.699 ms @100%
  C4: 1260×700×2                      2226          4.96        40.9%        73.066 ms @100%
  C5: 1280×768×15                    14910        183.55        49.6%      3278.066 ms @100%
  C6: 1260×700×2 mt512                2556          5.80        46.2%        96.335 ms @100%
==========================================================================================

========================================================================================================================
  Qwen3-Omni-4B — PA Kernel Comparison: FP16 vs INT8 BY_TOKEN vs INT8 BY_CHANNEL  (PTL 4Xe)
========================================================================================================================
  Case                               S        FP16 (NONE) (ms)     eff    INT8 (BY_TOKEN) (ms)     eff  INT8 (BY_CHANNEL) (ms)     eff
------------------------------------------------------------------------------------------------------------------------
  C1: 448×448×2                    902                   1.61   20.7%                   1.32   25.2%                   1.38   24.2%
  C2: 512×384×2                    894                   0.78   41.9%                   1.05   31.1%                   1.05   31.2%
  C3: 1024×512×2                  1534                   1.62   59.7%                   2.29   42.1%                   2.33   41.3%
  C4: 1260×700×2                  2226                   3.61   56.3%                   4.87   41.7%                   4.96   40.9%
  C5: 1280×768×15                14910                 120.08   75.8%                 180.16   50.5%                 183.55   49.6%
  C6: 1260×700×2 mt512            2556                   4.04   66.2%                   5.94   45.0%                   5.80   46.2%
========================================================================================================================