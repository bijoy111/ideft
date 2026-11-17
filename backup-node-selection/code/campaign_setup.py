"""Prepare the CloudLab nodes for campaign.py. Run this on your own machine.

    python3 campaign_setup.py nodes.txt            # check, install, copy, verify
    python3 campaign_setup.py nodes.txt --status   # how far the campaign has got
    python3 campaign_setup.py nodes.txt --fetch    # copy the results back here

nodes.txt is the experiment's List View pasted from the CloudLab page. Any line
that contains a node id (node0, node1, ...) and a hostname ending in cloudlab.us
is used; everything else is ignored. node0 becomes the client, node1..nodeN the
backup nodes.

Setup does, in order, and stops at the first thing that is wrong:
  1. reach every node over SSH
  2. check each node: 32 logical CPUs, cores 2-14 on distinct physical cores,
     passwordless sudo, tc, python3, and the experiment-LAN interface
  3. let node0 SSH to every backup node (campaign.py launches servers that way)
  4. copy this code directory to every node
  5. install Python packages (all nodes: numpy, scipy, networkx; node0 also
     torch, stable-baselines3, gymnasium for the PPO method)
  6. on node0: import the harness, load the 128-host PPO model, run the tests
  7. print the command that starts the campaign

It changes nothing on this machine and can be re-run safely.
"""

import argparse
import os
import re
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REMOTE_DIR = "ideft/backup-node-selection/code"       # under the remote home
BASE_PKGS = "numpy==2.2.6 scipy==1.15.3 networkx==3.1"
CLIENT_PKGS = "stable_baselines3==2.9.0 gymnasium==1.3.0"
TORCH = "torch==2.12.1"


def parse_nodes(path):
    found = {}
    for line in open(path):
        ids = re.findall(r"\bnode(\d+)\b", line)
        hosts = re.findall(r"[\w.-]+\.cloudlab\.us", line)
        if ids and hosts:
            found[int(ids[0])] = hosts[0]
    if not found:
        sys.exit("no lines with a node id and a *.cloudlab.us hostname in {}".format(path))
    missing = [i for i in range(max(found) + 1) if i not in found]
    if missing:
        sys.exit("node ids are not contiguous; missing: {}".format(
            ", ".join("node{}".format(i) for i in missing)))
    return [found[i] for i in range(len(found))]


class Cluster:
    def __init__(self, hosts, user, key):
        self.hosts, self.user, self.key = hosts, user, key
        self.base = ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                     "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30"]
        if key:
            self.base += ["-i", key]

    def target(self, i):
        return "{}@{}".format(self.user, self.hosts[i])

    def ssh(self, i, command, timeout=900, stdin=None):
        done = subprocess.run(self.base + [self.target(i), command], input=stdin,
                              capture_output=True, timeout=timeout)
        return done.returncode, done.stdout.decode(errors="replace").strip(), \
            done.stderr.decode(errors="replace").strip()

    def all(self, command, nodes=None, timeout=900):
        nodes = list(range(len(self.hosts))) if nodes is None else nodes
        with ThreadPoolExecutor(max_workers=len(nodes)) as pool:
            results = list(pool.map(lambda i: self.ssh(i, command, timeout), nodes))
        return dict(zip(nodes, results))

    def must(self, what, results, show=False):
        bad = {i: r for i, r in results.items() if r[0] != 0}
        if bad:
            print("FAILED: {}".format(what))
            for i, (code, out, err) in sorted(bad.items()):
                print("  node{} ({}): exit {}: {}".format(i, self.hosts[i], code,
                                                        (err or out)[-600:]))
            sys.exit(1)
        print("ok   {}".format(what))
        if show:
            for i, (_, out, _) in sorted(results.items()):
                print("       node{}: {}".format(i, out.splitlines()[-1] if out else ""))


CHECK = r"""
set -e
[ "$(nproc)" = 32 ] || { echo "expected 32 logical CPUs, found $(nproc)"; exit 1; }
cores=$(for c in $(seq 2 14); do cat /sys/devices/system/cpu/cpu$c/topology/core_id; done | sort -u | wc -l)
[ "$cores" = 13 ] || { echo "logical CPUs 2-14 cover $cores physical cores, expected 13"; exit 1; }
sudo -n true || { echo "no passwordless sudo"; exit 1; }
command -v tc >/dev/null || { echo "tc missing"; exit 1; }
command -v nstat >/dev/null || { echo "nstat missing"; exit 1; }
python3 -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || exit 1
iface=$(ip -o -4 addr show | awk '$4 ~ /^10\./ {print $2; exit}')
[ -n "$iface" ] || { echo "no interface with a 10.x address (experiment LAN)"; exit 1; }
getent hosts node0 | grep -q '^10\.' || { echo "node0 does not resolve to a LAN address"; exit 1; }
echo "$iface"
"""


def setup(c):
    n = len(c.hosts)
    backups = list(range(1, n))
    print("{} nodes: node0 is the client, node1..node{} carry the backups\n".format(n, n - 1))

    c.must("every node answers over SSH", c.all("true", timeout=60))
    checks = c.all(CHECK, timeout=120)
    c.must("CPUs, sudo, tc, python3 and experiment LAN on every node", checks)
    ifaces = {out.splitlines()[-1] for _, out, _ in checks.values()}
    if len(ifaces) != 1:
        sys.exit("the experiment-LAN interface differs between nodes: {}".format(
            {i: r[1].splitlines()[-1] for i, r in checks.items()}))
    iface = ifaces.pop()
    print("       experiment-LAN interface: {}".format(iface))

    code, pub, err = c.ssh(0, "test -f ~/.ssh/id_ed25519 || ssh-keygen -q -t ed25519 -N '' "
                              "-f ~/.ssh/id_ed25519; cat ~/.ssh/id_ed25519.pub")
    if code != 0 or not pub.startswith("ssh-ed25519"):
        sys.exit("could not create a key on node0: {}".format(err))
    add = ("mkdir -p ~/.ssh && chmod 700 ~/.ssh && touch ~/.ssh/authorized_keys && "
           "(grep -qxF {k} ~/.ssh/authorized_keys || echo {k} >> ~/.ssh/authorized_keys)"
           ).format(k=shlex.quote(pub.splitlines()[-1]))
    c.must("node0's key installed on every node", c.all(add, timeout=60))
    hop = " && ".join(
        "ssh -n -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=15 "
        "node{} true".format(i) for i in range(0, n))
    c.must("node0 can SSH to every node over the experiment LAN",
           {0: c.ssh(0, hop, timeout=300)})

    tar = subprocess.run(
        ["tar", "-C", HERE, "-czf", "-", "--exclude=results_rpc", "--exclude=__pycache__",
         "--exclude=venv", "--exclude=.git", "."], capture_output=True, check=True).stdout
    print("     copying the code ({:.1f} MB) to every node ...".format(len(tar) / 1e6))
    unpack = ("mkdir -p ~/{d} ~/{d}/../findings ~/{d}/results_rpc && "
              "tar -C ~/{d} -xzf -").format(d=REMOTE_DIR)
    with ThreadPoolExecutor(max_workers=n) as pool:
        copied = dict(zip(range(n), pool.map(
            lambda i: c.ssh(i, unpack, timeout=600, stdin=tar), range(n))))
    c.must("code copied to every node", copied)

    pip = ("(python3 -m pip --version >/dev/null 2>&1 || "
           "(sudo apt-get update -qq && sudo DEBIAN_FRONTEND=noninteractive "
           "apt-get install -y -qq python3-pip >/dev/null)) && "
           "python3 -m pip install --user -q {}".format(BASE_PKGS))
    print("     installing Python packages (a few minutes) ...")
    c.must("numpy, scipy, networkx installed on every node", c.all(pip, timeout=1800))
    client = ("sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq tmux >/dev/null; "
              "(python3 -m pip install --user -q {t} --index-url "
              "https://download.pytorch.org/whl/cpu || python3 -m pip install --user -q {t}) "
              "&& python3 -m pip install --user -q {p}".format(t=TORCH, p=CLIENT_PKGS))
    c.must("torch, stable-baselines3, gymnasium and tmux installed on node0",
           {0: c.ssh(0, client, timeout=3600)})

    c.must("backup server module imports on every backup node",
           c.all("cd ~/{} && python3 -c 'import backup_server'".format(REMOTE_DIR),
                 nodes=backups, timeout=120))
    sanity = (
        "cd ~/{d} && python3 test_telemetry.py >/dev/null && "
        "python3 -c \"import rl_backup_selector as r; "
        "r.load_rl_model('rl_model_scaled128_v3.zip', 'ppo'); print('ppo ok')\" && "
        "python3 campaign.py --nodes {n} --iface {i} --dry-run >/dev/null && "
        "(uname -a; lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|Core'; python3 --version; "
        "python3 -m pip freeze 2>/dev/null | grep -i -E "
        "'numpy|scipy|networkx|torch|stable|gymnasium') > ../findings/campaign_env.txt && "
        "echo ready").format(d=REMOTE_DIR, n=n - 1, i=iface)
    c.must("node0: tests pass, PPO model loads, campaign plan resolves",
           {0: c.ssh(0, sanity, timeout=900)})

    start = ("cd ~/{d} && tmux new -d -s campaign "
             "'python3 campaign.py --nodes {n} --iface {i} 2>&1 | tee -a ../findings/campaign.out'"
             ).format(d=REMOTE_DIR, n=n - 1, i=iface)
    print("\nAll nodes are ready. Start the campaign with:\n")
    print("  {} {} {}\n".format(" ".join(c.base), c.target(0), shlex.quote(start)))
    print("It runs inside tmux on node0 and keeps going if your connection drops.")
    print("Progress:  python3 campaign_setup.py {} --status".format(c.nodes_file))
    print("Results:   python3 campaign_setup.py {} --fetch".format(c.nodes_file))
    if n - 1 < 20:
        print("\nNote: {} backup nodes, so the 256-host steps (20 nodes) will be skipped.".format(n - 1))


def status(c):
    _, out, err = c.ssh(0, "cd ~/{}/../findings && (tmux has-session -t campaign 2>/dev/null "
                           "&& echo 'campaign: RUNNING' || echo 'campaign: not running'); "
                           "ls *.done 2>/dev/null | sed 's/.done$//' | tr '\\n' ' '; echo; "
                           "tail -n 12 campaign.log 2>/dev/null".format(REMOTE_DIR), timeout=60)
    print(out or err)


def fetch(c):
    local_findings = os.path.normpath(os.path.join(HERE, "..", "findings"))
    local_results = os.path.join(HERE, "results_rpc")
    os.makedirs(local_results, exist_ok=True)
    for remote, local in (("~/{}/../findings".format(REMOTE_DIR), local_findings),
                          ("~/{}/results_rpc".format(REMOTE_DIR), local_results)):
        print("fetching {} -> {}".format(remote, local))
        pull = subprocess.Popen(c.base + [c.target(0), "tar -C {} -czf - .".format(remote)],
                                stdout=subprocess.PIPE)
        subprocess.run(["tar", "-C", local, "-xzf", "-"], stdin=pull.stdout, check=True)
        if pull.wait() != 0:
            sys.exit("fetch failed for {}".format(remote))
    print("done. Raw records are in code/results_rpc/, reports in findings/.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("nodes_file", help="the CloudLab List View, pasted into a text file")
    ap.add_argument("--user", default="bijoy111", help="your CloudLab username")
    ap.add_argument("--key", default=os.path.expanduser("~/.ssh/cloudlab_ed25519"))
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--fetch", action="store_true")
    args = ap.parse_args()
    key = args.key if os.path.exists(args.key) else None
    c = Cluster(parse_nodes(args.nodes_file), args.user, key)
    c.nodes_file = args.nodes_file
    if args.status:
        return status(c)
    if args.fetch:
        return fetch(c)
    return setup(c)


if __name__ == "__main__":
    main()
