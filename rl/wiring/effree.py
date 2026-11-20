"""Effective free MiB on a GPU = nvidia-smi free - (library reservations not yet allocated).
RESERVE_g<gpu> files (written by lib_poller.sh): one line 'pid mem_gb' per library run pinned to that GPU."""
import os, sys, subprocess
gpu, wiring = sys.argv[1], sys.argv[2]
q = lambda a: subprocess.run(a, capture_output=True, text=True).stdout
free = int(q(['nvidia-smi', '-i', gpu, '--query-gpu=memory.free', '--format=csv,noheader,nounits']).split()[0])
apps = [l.split(',') for l in q(['nvidia-smi', '-i', gpu, '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits']).splitlines() if l.strip()]
apps = {int(p): int(m) for p, m in apps}
def ancestors(pid):
    out = set()
    while pid > 1:
        out.add(pid)
        try: pid = int(open(f'/proc/{pid}/stat').read().split(')')[1].split()[1])
        except Exception: break
    return out
pending = 0
f = os.path.join(wiring, f'RESERVE_g{gpu}')
if os.path.exists(f):
    for line in open(f):
        try: rp, mg = map(int, line.split())
        except ValueError: continue
        if not os.path.exists(f'/proc/{rp}'): continue
        used = sum(m for p, m in apps.items() if rp in ancestors(p))
        pending += max(0, mg * 1024 - used)
print(max(0, free - pending))
