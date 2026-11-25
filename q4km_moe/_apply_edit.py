import io, sys

path = "/mnt/river/kernel_foundry_master/workspace/q4km_moe/test_gemv_sg_kernels.py"
ins_path = "/mnt/river/kernel_foundry_master/workspace/q4km_moe/_q40q80_insert.py"

with open(path) as f:
    src = f.read()
with open(ins_path) as f:
    insert = f.read()

# 1) Insert the new functions right before the "Test cases" section.
anchor = ("# ---------------------------------------------------------------------------\n"
          "# Test cases\n")
assert anchor in src, "test-cases anchor not found"
src = src.replace(anchor, insert + "\n" + anchor, 1)

# 2) Add new test cases into TEST_CASES.
tc_anchor = "TEST_CASES = [\n"
assert tc_anchor in src, "TEST_CASES anchor not found"
new_cases = (
    tc_anchor +
    '    {"quant": "q4_0", "K": 12288, "N":   4096, "label": "Q4_0_SG_K12288_N4096"},\n'
    '    {"quant": "q8_0", "K": 12288, "N":   4096, "label": "Q8_0_SG_K12288_N4096"},\n'
    '    {"quant": "q4_0", "K":  4096, "N":   4096, "label": "Q4_0_SG_K4096_N4096"},\n'
    '    {"quant": "q8_0", "K":  4096, "N":   4096, "label": "Q8_0_SG_K4096_N4096"},\n'
    '    {"quant": "q4_0", "K":  4096, "N":   1024, "label": "Q4_0_SG_K4096_N1024"},\n'
    '    {"quant": "q8_0", "K":  4096, "N":   1024, "label": "Q8_0_SG_K4096_N1024"},\n'
)
src = src.replace(tc_anchor, new_cases, 1)

# 3) Replace the dispatch line in main() with a full mapping.
disp_old = '            fn  = bench_q4k_sg if case["quant"] == "q4k" else bench_q6k_sg\n'
assert disp_old in src, "dispatch anchor not found"
disp_new = (
    '            _DISPATCH = {"q4k": bench_q4k_sg,  "q6k": bench_q6k_sg,\n'
    '                         "q4_0": bench_q4_0_sg, "q8_0": bench_q8_0_sg}\n'
    '            fn  = _DISPATCH[case["quant"]]\n'
)
src = src.replace(disp_old, disp_new, 1)

with open(path, "w") as f:
    f.write(src)

print("edit applied OK")
