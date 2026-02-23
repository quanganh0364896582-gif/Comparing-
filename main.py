from utils.yolo11n_extract import YOLOLayerAnalyzer
from src.system_model import SystemModel
from src.lyapunov_partition import LyapunovPartition

import time
import matplotlib.pyplot as plt


# ===============================
# 1. Extract YOLOv11 layer info
# ===============================
analyzer = YOLOLayerAnalyzer("yolo11n.pt")
flops_layers, output_size_layers = analyzer.analyze()

# ===============================
# 2. TFLOPS scenarios
# ===============================
f_edge_set = {i: ((i + 1) % 10 + 1) * 2e12 for i in range(10)}
f_server_set = {i: ((i + 1) % 10 + 1) * 30e12 for i in range(10)}

bandwidth = 10  # MB/s

alpha = 0.0   # energy coefficient (scaled vì FLOPs rất lớn)
V = 0.0        # Lyapunov weight

# ===============================
# 3. Measure function
# ===============================
def measure(idx):
    system = SystemModel(
        f_edge=f_edge_set[idx],
        f_server=f_server_set[idx],
        bandwidth=bandwidth,
        alpha=alpha,
        V=V
    )

    optimizer = LyapunovPartition(
        flops_layers,
        output_size_layers,
        system
    )

    best_i, best_latency, best_energy, best_cost = optimizer.optimize()

    return best_i, best_latency, best_cost


# ===============================
# 4. Warm-up
# ===============================
for _ in range(10):
    measure(0)

# ===============================
# 5. Benchmark time
# ===============================
partition_results = []
latency_results = []
cost_results = []
time_results = []

for num in range(0, 20, 2):
    if num == 0:
        num = 1

    idx = num % 10

    n = 10
    lst_time = []

    for _ in range(n):
        start = time.perf_counter_ns()
        best_i, best_latency, best_cost = measure(idx)
        end = time.perf_counter_ns()
        lst_time.append(end - start)

    avg_time = sum(lst_time) / len(lst_time)
    time_results.append(avg_time)

    partition_results.append(best_i)
    latency_results.append(best_latency)
    cost_results.append(best_cost)

    print(f"{num} : {avg_time}")


# ===============================
# 6. Plot results
# ===============================

#plt.figure()
#plt.plot(partition_results, marker='o')
#plt.title("Optimal Partition Point vs Scenario")
#plt.xlabel("Scenario index")
#plt.ylabel("Partition layer index")
#plt.grid(True)


#plt.figure()
#plt.plot(latency_results, marker='o')
#plt.title("Latency vs Scenario")
#plt.xlabel("Scenario index")
#plt.ylabel("Latency (s)")
#plt.grid(True)


#plt.figure()
#plt.plot(cost_results, marker='o')
#plt.title("Lyapunov Cost vs Scenario")
#plt.xlabel("Scenario index")
#plt.ylabel("Cost")
#plt.grid(True)

#plt.show()
