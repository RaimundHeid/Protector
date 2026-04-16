import os
import subprocess
import re
import time
import math
import sys

# SPSA parameters (Stockfish-style)
# theta: starting value of the constant to optimize (malus = 128 * bonus / 64)
# c_init: initial perturbation = 15% of theta (wide margin for first iteration)
# gamma: controls how c shrinks: c_k = c_init / k^gamma
# alpha: controls how a shrinks: a_k = a_init / (A + k)^alpha
# A: stabilization constant (~10% of expected iterations)
alpha = 0.602
gamma = 0.101
A = 10
theta = 128.0
c_init = 128.0 * 0.15   # = 19.2  (±15% of starting value)
a_init = 50.0            # step size constant (scaled for theta=128)

CUTECHESS_LOG = "/tmp/cutechess_spsa.log"
SPSA_LOG      = "/tmp/spsa_tune.log"

def update_file(new_theta):
    with open("search.c", "r") as f:
        content = f.read()
    # Targets: const int malus = 128 * bonus / 64;
    pattern = r'(const int malus = )\d+( \* bonus / 64)'
    new_content = re.sub(pattern, r'\g<1>' + str(int(round(new_theta))) + r'\2', content)
    with open("search.c", "w") as f:
        f.write(new_content)

def run_match(iteration):
    with open(CUTECHESS_LOG, "w") as f:
        f.write("Starting match (iteration {})...\n".format(iteration))

    print(f"  Monitor cutechess output with:")
    print(f"    tail -n100 -f {CUTECHESS_LOG}")
    sys.stdout.flush()

    cmd = "time runMatch.sh proLow 10+0.1 proHigh 10+0.1 standard 250 >> " + CUTECHESS_LOG + " 2>&1"
    subprocess.run(cmd, shell=True)

    with open(CUTECHESS_LOG, "r") as f:
        lines = f.readlines()

    score_line = next((l for l in reversed(lines) if l.strip().startswith("Score of proLow vs proHigh:")), None)
    elo_line   = next((l for l in reversed(lines) if l.strip().startswith("Elo difference:")), None)

    if score_line:
        print(f"  {score_line.strip()}")
    if elo_line:
        print(f"  {elo_line.strip()}")
        elo_match = re.search(r"Elo difference:\s*([-\d.]+)", elo_line)
        if elo_match:
            return float(elo_match.group(1))
    return 0.0

print("Starting SPSA optimization for constant 128 in search.c line 1058 (malus)")
print(f"  theta_0 = {theta:.2f}, c_init = {c_init:.2f}, a_init = {a_init:.2f}")
print(f"  Monitor this script's output with:")
print(f"    tail -n100 -f {SPSA_LOG}")
sys.stdout.flush()

iteration = 0

while True:
    ck = c_init / (iteration + 1)**gamma
    ak = a_init / (A + iteration + 1)**alpha

    theta_low  = max(1.0, theta - ck)
    theta_high = theta + ck

    print(f"\nIteration {iteration}: theta = {theta:.2f}, c = {ck:.2f}, a = {ak:.2f}")
    print(f"  Low = {theta_low:.2f}, High = {theta_high:.2f}")
    sys.stdout.flush()

    # Build Low version
    print("  Building proLow...")
    sys.stdout.flush()
    update_file(theta_low)
    subprocess.run("buildLow.sh", shell=True, check=True, capture_output=True)

    # Build High version
    print("  Building proHigh...")
    sys.stdout.flush()
    update_file(theta_high)
    subprocess.run("buildHigh.sh", shell=True, check=True, capture_output=True)

    # Restore search.c to current best theta before running match
    update_file(theta)
    print(f"  Restored search.c to current best theta = {int(round(theta))}")
    sys.stdout.flush()

    # Run match
    elo = run_match(iteration)
    print(f"  Elo proLow vs proHigh: {elo:.2f}")
    sys.stdout.flush()

    # SPSA gradient update: negative elo means proLow won → lower value is better
    theta_update = -ak * (elo / (2.0 * ck))
    theta += theta_update
    print(f"  Update: {theta_update:.4f}, New theta: {theta:.2f} → rounded: {int(round(theta))}")
    sys.stdout.flush()

    if iteration > 0 and abs(elo) <= 4.0:
        print(f"\nConvergence reached: |Elo diff| = {abs(elo):.2f} <= 4.0")
        update_file(theta)
        print(f"Final value written to search.c: {int(round(theta))}")
        sys.stdout.flush()
        break

    iteration += 1
