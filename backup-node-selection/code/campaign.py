"""Run the whole bare-metal campaign, one step after another, on the client node.

Each step provisions the backup servers it needs, checks that every server
answers on a core of its own, imposes the step's degradation with tc, runs the
experiment, clears the degradation and stops the servers. A step that finished is
skipped when the script is started again, so an interrupted campaign resumes where
it stopped.

Run it on the client node (node0), inside tmux so it survives a dropped SSH
session. campaign_setup.sh prepares the nodes and prints the exact command.

    python3 campaign.py --nodes 20 --iface enp65s0f0np0 --list      # the plan
    python3 campaign.py --nodes 20 --iface enp65s0f0np0 --dry-run   # the commands
    python3 campaign.py --nodes 20 --iface enp65s0f0np0             # run it
    python3 campaign.py --nodes 20 --iface enp65s0f0np0 --only s256_binary

`--nodes N` is the number of backup nodes, named node1..nodeN on the experiment
LAN. Steps that need more nodes than are present are skipped with a note, so the
same command works on an 11-node experiment (no 256-host steps).

`--smoke` runs a miniature of every kind of step on this machine with local
servers and no tc, to test the script itself. Its numbers mean nothing.

Outputs, per step <name>:
    ../findings/<name>.txt             the run's full report
    ../findings/<name>.counters.txt    kernel listen-overflow counters before/after
    ../findings/manifest_<name>.json   where each server ran, and on which core
    results_rpc/<name>.csv             raw per-call records
"""

import argparse
import os
import subprocess
import sys
import time

FINDINGS = os.path.join("..", "findings")
RESULTS = "results_rpc"
PER_NODE = 13          # servers per backup node; logical CPUs 2-14 are 13 physical cores
BACKLOG = 512
CLIENT_PROCS = 16      # the client node has 16 physical cores

# hosts -> (primaries, degraded, backup nodes used, servers per node, PPO checkpoint).
# 16 and 32 hosts keep the October layout (8 nodes, 2 and 4 servers each).
SCALES = {
    16: (10, 2, 8, 2, "rl_model_scaled.zip"),
    32: (20, 4, 8, 4, "rl_model_scaled32.zip"),
    128: (80, 16, 10, PER_NODE, "rl_model_scaled128_v3.zip"),
    256: (160, 32, 20, PER_NODE, None),
    # 19 backup nodes x 13 servers, for an experiment with one faulty node.
    # Primaries and degraded hosts keep the 5/8 and 1/8 shares, rounded.
    247: (154, 31, 19, PER_NODE, None),
}
BASE_METHODS = ["concentrated", "random", "round_robin", "heuristic", "optimal"]


LAYOUTS = 5            # degraded layouts per degraded configuration


def step(name, hosts, seeds, model=None, client_procs=CLIENT_PROCS, probe=False,
         rtt_noise=0.0, feature_noise=0.0, stale=0.0, note="", layouts=None):
    """One configuration. A configuration with degraded hosts is run under several
    layouts (see expand): which hosts are degraded, and under the realistic model
    their failure kinds and severities, are part of the problem instance, and one
    layout measured 30 times is one instance of it."""
    if layouts is None:
        layouts = LAYOUTS if (model and seeds >= LAYOUTS) else 1
    return dict(name=name, hosts=hosts, seeds=seeds, model=model,
                client_procs=client_procs, probe=probe, rtt_noise=rtt_noise,
                feature_noise=feature_noise, stale=stale, note=note, layouts=layouts,
                seed_lo=1, selection="spread", scale_seed=0)


def expand(steps):
    """Split each configuration into one sub-step per layout.

    Layout 0 is the October one (degraded hosts spread evenly, scale seed 0), so
    the new runs include the old configuration. Layouts 1.. draw the degraded
    hosts at random with scale seed = layout number. The context seeds are divided
    between the layouts, so every seed is still a distinct problem instance."""
    out = []
    for s in steps:
        if s["layouts"] == 1:
            out.append(s)
            continue
        per = s["seeds"] // s["layouts"]
        for k in range(s["layouts"]):
            sub = dict(s, name="{}_L{}".format(s["name"], k), seeds=per,
                       seed_lo=1 + k * per, scale_seed=k,
                       selection="spread" if k == 0 else "random", layouts=1)
            out.append(sub)
    return out


# In priority order: if the reservation runs out, what is lost is the tail.
STEPS = [
    # Preflight: a two-seed run with tc and the probe. The campaign stops if the
    # probe does not see the degraded hosts slow, since nothing after it would
    # mean anything.
    step("preflight_16", 16, 2, model="binary", probe=True,
         note="preflight: is the degradation real and visible?"),
    # 0. Does the client itself enter the measurement? Same configuration, one
    #    client process (as in October) against CLIENT_PROCS.
    step("check_client1_128_homog", 128, 5, client_procs=1,
         note="client check: one client process, as in October"),
    step("check_client16_128_homog", 128, 5,
         note="client check: multi-process client"),
    # 1. The October configuration with 30 seeds, multi-process client.
    step("s16_binary", 16, 30, model="binary"),
    step("s32_binary", 32, 30, model="binary"),
    step("s128_binary", 128, 30, model="binary"),
    # 2. Homogeneous negative control.
    step("s128_homog", 128, 30),
    # 3. Realistic degradation, selectors fed probed RTT.
    step("s32_realistic_probe", 32, 30, model="realistic", probe=True),
    step("s128_realistic_probe", 128, 30, model="realistic", probe=True),
    # 4. The same with noisy and partly stale telemetry.
    step("s128_realistic_noisy", 128, 30, model="realistic", probe=True,
         rtt_noise=2.0, feature_noise=0.05, stale=0.25),
    # 5-6. 256 hosts.
    step("s256_binary", 256, 30, model="binary"),
    step("s256_realistic_probe", 256, 30, model="realistic", probe=True),
    step("s256_homog", 256, 30),
    # The same at 247 hosts, the largest that 19 backup nodes carry.
    step("s247_binary", 247, 30, model="binary"),
    step("s247_realistic_probe", 247, 30, model="realistic", probe=True),
    step("s247_homog", 247, 30),
    # 7. October's exact client (one process) at 30 seeds, so the change of client
    #    can be separated from the change of seed count.
    step("s16_binary_client1", 16, 30, model="binary", client_procs=1),
    step("s32_binary_client1", 32, 30, model="binary", client_procs=1),
    step("s128_binary_client1", 128, 30, model="binary", client_procs=1),
]

SMOKE_STEPS = [
    step("smoke_homog_client1", 16, 2, client_procs=1),
    step("smoke_binary", 16, 4, model="binary", client_procs=2, layouts=2),
    step("smoke_realistic_noisy", 16, 2, model="realistic", client_procs=2, probe=True,
         rtt_noise=1.0, feature_noise=0.05, stale=0.5),
]


class Runner:
    def __init__(self, args):
        self.args = args
        self.log = None

    def say(self, text):
        line = "[{}] {}".format(time.strftime("%H:%M:%S"), text)
        print(line, flush=True)
        if self.log:
            self.log.write(line + "\n")
            self.log.flush()

    def run(self, argv, check=True, capture=False, quiet=False):
        """Run one command; in --dry-run only print it."""
        shown = " ".join(argv)
        if self.args.dry_run:
            print("    " + shown)
            return ""
        if not quiet:
            self.say("$ " + shown)
        done = subprocess.run(argv, text=True, stdout=subprocess.PIPE if capture else None,
                              stderr=subprocess.STDOUT if capture else None)
        if check and done.returncode != 0:
            raise RuntimeError("command failed ({}): {}\n{}".format(
                done.returncode, shown, done.stdout or ""))
        return done.stdout or ""

    # -- pieces of a step ---------------------------------------------------

    def usable_nodes(self):
        return [i for i in range(1, self.args.nodes + 1) if i not in self.args.exclude_nodes]

    def nodes_for(self, hosts):
        n_nodes = SCALES[hosts][2]
        return ["node{}".format(i) for i in self.usable_nodes()[:n_nodes]]

    def scenario_flags(self, s):
        primaries, n_degraded = SCALES[s["hosts"]][:2]
        return ["--n-hosts", str(s["hosts"]), "--n-primaries", str(primaries),
                "--n-degraded", str(n_degraded if s["model"] else 0),
                "--scale-seed", str(s["scale_seed"])]

    def counters(self, s, when):
        if self.args.smoke or self.args.dry_run:
            return
        path = os.path.join(FINDINGS, s["name"] + ".counters.txt")
        with open(path, "a") as out:
            out.write("=== {} {} ===\n".format(when, time.strftime("%Y-%m-%d %H:%M:%S")))
            for node in self.nodes_for(s["hosts"]):
                done = subprocess.run(
                    ["ssh", "-n", "-o", "BatchMode=yes", node,
                     "nstat -az TcpExtListenOverflows TcpExtListenDrops TcpRetransSegs "
                     "| tail -n +2 | awk '{print $1, $2}' | tr '\\n' ' '"],
                    text=True, capture_output=True)
                out.write("{:<8} {}\n".format(node, done.stdout.strip() or done.stderr.strip()))

    def provision(self, s, manifest):
        hosts = s["hosts"]
        if self.args.smoke:
            self.run([sys.executable, "provision_remote.py", "--local", "--hosts", str(hosts),
                      "--per-node", str(hosts), "--backlog", str(BACKLOG),
                      "--manifest", manifest], quiet=True, capture=True)
        else:
            self.run([sys.executable, "provision_remote.py",
                      "--nodes"] + self.nodes_for(hosts) + [
                      "--hosts", str(hosts), "--per-node", str(SCALES[hosts][3]),
                      "--reserve-cores", "2", "--backlog", str(BACKLOG),
                      "--remote-dir", os.getcwd(), "--manifest", manifest],
                     capture=True)
        out = self.run([sys.executable, "provision_remote.py", "--manifest", manifest,
                        "--verify"], capture=True)
        if not self.args.dry_run:
            if "WARNING" in out and not self.args.smoke:
                raise RuntimeError("server verification warned:\n" + out)
            self.say("servers verified: " + (out.strip().splitlines() or ["?"])[-1].strip())

    def degrade(self, s, manifest, apply_it):
        if self.args.smoke or not s["model"]:
            return
        base = [sys.executable, "apply_degradation.py", "--manifest", manifest,
                "--iface", self.args.iface]
        if not apply_it:
            self.run(base + ["--clear"], capture=True, check=False)
            return
        out = self.run(base + ["--scenario"] + self.scenario_flags(s) + [
            "--degradation-model", s["model"], "--degraded-selection", s["selection"],
            "--apply"], capture=True)
        if not self.args.dry_run:
            with open(os.path.join(FINDINGS, s["name"] + ".degradation.txt"), "w") as f:
                f.write(out)
            shown = self.run(base + ["--show"], capture=True, check=False)
            n_netem = shown.count("netem")
            self.say("degradation applied; netem qdiscs live: {}".format(n_netem))
            if n_netem == 0:
                raise RuntimeError("no netem qdisc found after applying degradation")

    def experiment(self, s, manifest):
        ppo = SCALES[s["hosts"]][4]
        methods = BASE_METHODS + (["rl"] if ppo and not self.args.smoke else [])
        calls, burst, trials = ("20", "5", "1") if self.args.smoke else ("200", "10", "3")
        argv = [sys.executable, "rpc_contention_experiment.py",
                "--context-seeds"] + [str(i) for i in range(s["seed_lo"], s["seed_lo"] + s["seeds"])] + [
                "--trials", trials, "--calls", calls, "--burst", burst,
                "--methods"] + methods + self.scenario_flags(s) + [
                "--manifest", manifest, "--client-procs", str(s["client_procs"]),
                "--label", s["name"],
                "--results-csv", os.path.join(RESULTS, s["name"] + ".csv")]
        if ppo and not self.args.smoke:
            argv += ["--rl-model", ppo]
        if s["model"]:
            argv += ["--degradation-model", s["model"], "--degraded-selection", s["selection"],
                     "--physical-degradation"]
        if s["probe"]:
            argv += ["--rtt-probe"]
        for flag, key in (("--rtt-noise-ms", "rtt_noise"), ("--feature-noise", "feature_noise"),
                          ("--stale-frac", "stale")):
            if s[key]:
                argv += [flag, str(s[key])]
        if self.args.dry_run:
            self.run(argv)
            return
        self.say("$ " + " ".join(argv))
        with open(os.path.join(FINDINGS, s["name"] + ".stdout.txt"), "w") as sink:
            done = subprocess.run(argv, stdout=sink, stderr=subprocess.STDOUT)
        if done.returncode != 0:
            raise RuntimeError("experiment failed; see findings/{}.stdout.txt".format(s["name"]))

    # -- a whole step ---------------------------------------------------------

    def do(self, s):
        name = s["name"]
        manifest = os.path.join(FINDINGS, "manifest_{}.json".format(name))
        marker = os.path.join(FINDINGS, name + ".done")
        if os.path.exists(marker) and not self.args.redo:
            self.say("skip {} (already done)".format(name))
            return "skipped"
        if not self.args.smoke and SCALES[s["hosts"]][2] > len(self.usable_nodes()):
            self.say("skip {}: needs {} backup nodes, {} usable".format(
                name, SCALES[s["hosts"]][2], len(self.usable_nodes())))
            return "skipped"
        self.say("=== {} === {} hosts, seeds {}-{}, model={}, layout={}/{}, client procs={} {}".format(
            name, s["hosts"], s["seed_lo"], s["seed_lo"] + s["seeds"] - 1, s["model"] or "none",
            s["selection"], s["scale_seed"], s["client_procs"], s["note"]))
        started = time.time()
        try:
            self.provision(s, manifest)
            self.degrade(s, manifest, apply_it=True)
            self.counters(s, "before")
            self.experiment(s, manifest)
            self.counters(s, "after")
        finally:
            # Always leave the nodes clean, also after a failure or Ctrl-C.
            try:
                self.degrade(s, manifest, apply_it=False)
                if os.path.exists(manifest) or self.args.dry_run:
                    self.run([sys.executable, "provision_remote.py", "--manifest", manifest,
                              "--teardown"], check=False, capture=True, quiet=True)
            except Exception as exc:  # cleanup must not hide the original error
                self.say("cleanup problem: {}".format(exc))
        if not self.args.dry_run:
            minutes = (time.time() - started) / 60.0
            with open(marker, "w") as f:
                f.write("{:.1f} minutes\n".format(minutes))
            self.say("done {} in {:.1f} min".format(name, minutes))
        return "done"

    def preflight_ok(self, s):
        """The probe must have measured both degraded hosts well above the rest."""
        path = os.path.join(FINDINGS, s["name"] + ".txt")
        excess = []
        with open(path) as f:
            for line in f:
                if "(degraded)" in line and "excess" in line:
                    excess.append(float(line.split("excess")[1].split("ms")[0]))
        expected = SCALES[s["hosts"]][1]
        self.say("preflight: probe excess on degraded hosts: {} ms".format(
            ", ".join("{:.1f}".format(e) for e in excess) or "none found"))
        return len(excess) == expected and min(excess) > 50.0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nodes", type=int, default=0,
                    help="number of backup nodes (node1..nodeN); the client is node0")
    ap.add_argument("--exclude-nodes", type=int, nargs="+", default=[],
                    help="backup node numbers to leave out (a faulty node)")
    ap.add_argument("--iface", default=None, help="experiment-LAN interface on the nodes")
    ap.add_argument("--list", action="store_true", help="print the plan and exit")
    ap.add_argument("--dry-run", action="store_true", help="print every command, run none")
    ap.add_argument("--only", nargs="+", default=None, help="run just these steps")
    ap.add_argument("--skip", nargs="+", default=[], help="leave these steps out")
    ap.add_argument("--redo", action="store_true", help="re-run steps already marked done")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny local run of each kind of step, to test this script")
    args = ap.parse_args()

    steps = expand(SMOKE_STEPS if args.smoke else STEPS)
    names = [s["name"] for s in steps]
    def chosen(name, wanted):          # a configuration name selects all its layouts
        return any(name == w or name.startswith(w + "_L") for w in wanted)
    for wanted in (args.only or []) + args.skip:
        if not any(chosen(n, [wanted]) for n in names):
            ap.error("unknown step {!r}; choose from: {}".format(wanted, ", ".join(names)))
    steps = [s for s in steps
             if (not args.only or chosen(s["name"], args.only))
             and not chosen(s["name"], args.skip)]

    if args.list:
        for s in steps:
            needs = SCALES[s["hosts"]][2]
            print("{:<28} {:>3} hosts seeds {:>2}-{:<2} model={:<9} layout={:<6}/{} client={:<2} nodes={:<2} {}".format(
                s["name"], s["hosts"], s["seed_lo"], s["seed_lo"] + s["seeds"] - 1,
                s["model"] or "none", s["selection"] if s["model"] else "-", s["scale_seed"],
                s["client_procs"], needs, "done" if os.path.exists(os.path.join(FINDINGS, s["name"] + ".done")) else ""))
        return 0
    if not args.smoke and (args.nodes < 1 or not args.iface):
        ap.error("--nodes and --iface are required (campaign_setup.sh prints them)")

    os.makedirs(FINDINGS, exist_ok=True)
    os.makedirs(RESULTS, exist_ok=True)
    runner = Runner(args)
    if not args.dry_run:
        runner.log = open(os.path.join(FINDINGS, "campaign_smoke.log" if args.smoke
                                       else "campaign.log"), "a")
    failed = []
    for s in steps:
        try:
            status = runner.do(s)
            if (s["name"].startswith("preflight") and status == "done"
                    and not args.dry_run and not runner.preflight_ok(s)):
                os.remove(os.path.join(FINDINGS, s["name"] + ".done"))
                runner.say("STOP: the probe did not find the degraded hosts about 100 ms "
                           "slow. Check --iface and findings/{}.degradation.txt.".format(s["name"]))
                return 2
        except KeyboardInterrupt:
            runner.say("interrupted during {}; nodes were cleaned up".format(s["name"]))
            return 130
        except Exception as exc:
            # One failed step must not cost the rest of the reservation.
            failed.append(s["name"])
            runner.say("FAILED {}: {}".format(s["name"], str(exc)[:2000]))
    runner.say("campaign finished; failed steps: {}".format(", ".join(failed) or "none"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
