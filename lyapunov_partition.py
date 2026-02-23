import numpy as np


class LyapunovPartition:
    def __init__(self, flops_layers, output_size_layers, system: 'SystemModel'):
        self.F = np.array(flops_layers, dtype=float)
        self.S = np.array(output_size_layers, dtype=float)
        self.sys = system

        self.L = len(self.F)

        # Prefix FLOPs for energy
        self.edge_flops_prefix = np.cumsum(self.F)

        # Compute prefix/suffix time
        self.edge_prefix = self.edge_flops_prefix / self.sys.f_edge

        total_server_flops = np.sum(self.F)
        self.server_suffix = (total_server_flops - self.edge_flops_prefix) / self.sys.f_server

    def optimize(self):
        best_i = 0
        best_latency = float("inf")
        best_energy = float("inf")
        best_cost = float("inf")

        for i in range(self.L):

            T_edge = self.edge_prefix[i]
            T_server = self.server_suffix[i]

            T_tx = self.S[i] / self.sys.bandwidth if i < len(self.S) else 0

            latency = max(T_edge, T_server) + T_tx

            energy = self.sys.alpha * self.edge_flops_prefix[i]

            cost = latency + self.sys.V * energy

            if cost < best_cost:
                best_cost = cost
                best_latency = latency
                best_energy = energy
                best_i = i

        return best_i, best_latency, best_energy, best_cost
