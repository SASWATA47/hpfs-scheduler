import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------------
# CONFIGURATION
# -----------------------------------
NUM_NODES = 4
NODE_CAPACITY = 2000  # millicores
NUM_PODS = 40
SIM_TIME = 10  # minutes
ARRIVAL_RATE = NUM_PODS / SIM_TIME  # pods per minute

# Pod structure: (id, cpu_req, duration, tenant)
pods = [
    {
        "id": f"pod-{i+1}",
        "cpu_req": random.randint(100, 800),
        "duration": random.randint(1, 5),
        "tenant": random.choice(["T1", "T2", "T3"]),
    }
    for i in range(NUM_PODS)
]

# -----------------------------------
# Helper functions
# -----------------------------------
def jain_fairness_index(x):
    x = np.array(x)
    return (np.sum(x) ** 2) / (len(x) * np.sum(x ** 2))

def avg_cpu_variance(utilizations):
    mean = np.mean(utilizations)
    return np.mean([(u - mean) ** 2 for u in utilizations])

def throughput(completed_pods, total_time):
    return completed_pods / total_time

# -----------------------------------
# Default Scheduler (Bin Packing)
# -----------------------------------
def default_scheduler(pods, node_capacity=NODE_CAPACITY, num_nodes=NUM_NODES):
    nodes = [{"id": f"node-{i+1}", "used": 0, "pods": []} for i in range(num_nodes)]
    pending = 0
    time = 0
    completed = 0
    active_pods = []

    for pod in pods:
        scheduled = False
        for node in nodes:
            if node["used"] + pod["cpu_req"] <= node_capacity:
                node["pods"].append(pod)
                node["used"] += pod["cpu_req"]
                active_pods.append(pod)
                scheduled = True
                break
        if not scheduled:
            pending += 1

    # Simulate completion
    completed = len(active_pods)
    utilizations = [n["used"] / node_capacity for n in nodes]
    fairness = jain_fairness_index(utilizations)
    variance = avg_cpu_variance(utilizations)
    thr = throughput(completed, SIM_TIME)
    return variance, fairness, pending, thr, utilizations

# -----------------------------------
# HPFS Scheduler
# -----------------------------------
def hpfs_scheduler(pods, node_capacity=NODE_CAPACITY, num_nodes=NUM_NODES):
    nodes = [{"id": f"node-{i+1}", "used": 0, "pods": [], "ema": 0} for i in range(num_nodes)]
    pending = 0
    completed = 0

    # weights
    wR, wB, wP, wA, wAge = 0.35, 0.25, 0.2, 0.1, 0.1
    alpha = 0.6  # EMA smoothing factor

    for time, pod in enumerate(pods, 1):
        best_node = None
        best_score = -1
        for node in nodes:
            free = node_capacity - node["used"]
            if free < pod["cpu_req"]:
                continue

            # Compute metrics
            resource_fit = pod["cpu_req"] / free
            balance = 1 - (node["used"] / node_capacity)
            ema = alpha * balance + (1 - alpha) * node["ema"]
            affinity = 1 if any(p["tenant"] == pod["tenant"] for p in node["pods"]) else 0
            aging = min(1, time / NUM_PODS)  # grows with time

            score = (
                wR * resource_fit
                + wB * balance
                + wP * ema
                + wA * affinity
                + wAge * aging
            )

            if score > best_score:
                best_score = score
                best_node = node

        if best_node:
            best_node["pods"].append(pod)
            best_node["used"] += pod["cpu_req"]
            best_node["ema"] = alpha * (1 - best_node["used"] / node_capacity) + (1 - alpha) * best_node["ema"]
            completed += 1
        else:
            pending += 1

    utilizations = [n["used"] / node_capacity for n in nodes]
    fairness = jain_fairness_index(utilizations)
    variance = avg_cpu_variance(utilizations)
    thr = throughput(completed, SIM_TIME)
    return variance, fairness, pending, thr, utilizations

# -----------------------------------
# Run Both Schedulers
# -----------------------------------
hpfs_results = hpfs_scheduler(pods)
default_results = default_scheduler(pods)

# Create DataFrame for comparison
df = pd.DataFrame({
    "Metric": ["Avg CPU Variance", "Jain Fairness Index", "Pending Pods", "Throughput (pods/min)"],
    "HPFS": [hpfs_results[0], hpfs_results[1], hpfs_results[2], hpfs_results[3]],
    "Default": [default_results[0], default_results[1], default_results[2], default_results[3]],
})

# Compute improvement
def improvement(a, b, higher_better=False):
    if higher_better:
        return ((a - b) / b) * 100
    else:
        return ((b - a) / b) * 100

df["Improvement (%)"] = [
    improvement(df["HPFS"][0], df["Default"][0]),
    improvement(df["HPFS"][1], df["Default"][1], higher_better=True),
    "Eliminated" if df["Default"][2] > df["HPFS"][2] else "—",
    improvement(df["HPFS"][3], df["Default"][3], higher_better=True)
]

print("\n=== HPFS vs Default Scheduler ===\n")
print(df.to_string(index=False))

# -----------------------------------
# Visualization
# -----------------------------------
metrics = ["Avg CPU Variance", "Jain Fairness Index", "Throughput (pods/min)"]
for i, metric in enumerate(metrics):
    plt.figure()
    plt.bar(["HPFS", "Default"], [df.loc[i, "HPFS"], df.loc[i, "Default"]],
            color=["orange", "gray"])
    plt.title(metric)
    plt.ylabel(metric)
    plt.show()