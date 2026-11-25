#!/usr/bin/env python3
"""Remote driver for the MoE 3-GEMM SwiGLU kernel optimization on Windows machine-3 (Arc B390).

Uploads the harness tree + current kernel + config to the remote, runs moe_bench.py with the
venv python (has pyopencl), fetches the JSON report back, and returns parsed per-case results.

CLI:
  python3 remote_run.py setup            # upload full tree (harness/kernels/configs)
  python3 remote_run.py gpu              # print remote OpenCL platforms/devices
  python3 remote_run.py bench [cfg.json] # upload kernel+config, run bench, print + fetch report
  python3 remote_run.py raw "cmd"        # run an arbitrary cmd in the remote dir
"""
import sys, os, json
import paramiko

HOST = "10.239.132.229"
USER = "Local_Admin"
PWD  = "openvino"
REMOTE_DIR       = r"D:\river\ovmx\moe_kernel"
REMOTE_DIR_POSIX = "D:/river/ovmx/moe_kernel"
PY = r"C:\Users\Local_Admin\river\py310\Scripts\python.exe"

# mapelites/ lives inside moe_kernel/, so the package root is the parent dir.
LOCAL_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL_REL = "kernels/moe_3gemm_swiglu_mlp.cl"
UPLOAD_TREES = ["harness", "kernels", "configs"]


def connect():
    c = paramiko.SSHClient()
    c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    c.connect(HOST, username=USER, password=PWD, timeout=30,
              look_for_keys=False, allow_agent=False)
    return c


def run(c, cmd, cd=True):
    full = ('cd /d "%s" && %s' % (REMOTE_DIR, cmd)) if cd else cmd
    _, stdout, stderr = c.exec_command(full, timeout=3600)
    out = stdout.read().decode(errors="replace")
    err = stderr.read().decode(errors="replace")
    rc = stdout.channel.recv_exit_status()
    return rc, out, err


def _mkdirs(sftp, remote_posix):
    parts = remote_posix.split("/")
    cur = parts[0]
    for p in parts[1:]:
        cur += "/" + p
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur)


def put(c, local_rel, remote_rel=None):
    remote_rel = remote_rel or local_rel
    sftp = c.open_sftp()
    lp = os.path.join(LOCAL_DIR, local_rel)
    rp = REMOTE_DIR_POSIX + "/" + remote_rel.replace("\\", "/")
    _mkdirs(sftp, os.path.dirname(rp))
    sftp.put(lp, rp)
    sftp.close()
    return rp


def put_tree(c, subdir):
    sftp = c.open_sftp()
    base = os.path.join(LOCAL_DIR, subdir)
    n = 0
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("__pycache__", ".git", "_variants")]
        for f in files:
            if f.endswith(".pyc"):
                continue
            lp = os.path.join(root, f)
            rel = os.path.relpath(lp, LOCAL_DIR).replace("\\", "/")
            rp = REMOTE_DIR_POSIX + "/" + rel
            _mkdirs(sftp, os.path.dirname(rp))
            sftp.put(lp, rp)
            n += 1
    sftp.close()
    return n


def get(c, remote_rel, local_rel=None):
    local_rel = local_rel or remote_rel
    sftp = c.open_sftp()
    rp = REMOTE_DIR_POSIX + "/" + remote_rel.replace("\\", "/")
    lp = os.path.join(LOCAL_DIR, local_rel)
    os.makedirs(os.path.dirname(lp), exist_ok=True)
    sftp.get(rp, lp)
    sftp.close()
    return lp


def bench(cfg_name="qwen36.json", kernel_rel=KERNEL_REL):
    """Upload the current kernel + config, run moe_bench, return (rc, out, err, report_dict|None)."""
    c = connect()
    try:
        sftp = c.open_sftp(); _mkdirs(sftp, REMOTE_DIR_POSIX); sftp.close()
        put(c, kernel_rel)
        put(c, "configs/%s" % cfg_name)
        res_name = cfg_name.replace(".json", "_report.json")
        rc, out, err = run(c, '%s harness\\moe_bench.py --config configs\\%s --out results\\%s'
                           % (PY, cfg_name, res_name))
        report = None
        try:
            lp = get(c, "results/%s" % res_name, "mapelites/_last_report.json")
            with open(lp) as f:
                report = json.load(f)
        except Exception:
            pass
        return rc, out, err, report
    finally:
        c.close()


def main():
    if len(sys.argv) < 2:
        print(__doc__); return
    cmd = sys.argv[1]
    c = connect()
    try:
        sftp = c.open_sftp(); _mkdirs(sftp, REMOTE_DIR_POSIX); sftp.close()
        if cmd == "setup":
            for t in UPLOAD_TREES:
                print("uploaded %d files from %s/" % (put_tree(c, t), t))
        elif cmd == "gpu":
            code = ("import pyopencl as cl;"
                    "[print(p.name,'::',[d.name for d in p.get_devices()]) for p in cl.get_platforms()]")
            rc, out, err = run(c, '%s -c "%s"' % (PY, code))
            print("rc=", rc); print(out); print(err)
        elif cmd == "raw":
            rc, out, err = run(c, sys.argv[2])
            print("rc=", rc); print(out); print(err)
        elif cmd == "bench":
            cfg = sys.argv[2] if len(sys.argv) > 2 else "qwen36.json"
            c.close()
            rc, out, err, rep = bench(cfg)
            print("=== BENCH rc=%d ===" % rc)
            print(out)
            if err.strip():
                print("--- stderr ---"); print(err[-4000:])
            print("report parsed:", bool(rep))
            return
        else:
            print("unknown cmd", cmd)
    finally:
        try:
            c.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
