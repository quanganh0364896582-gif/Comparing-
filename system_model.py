class SystemModel:
    def __init__(self, f_edge, f_server, bandwidth, alpha=1e-12, V=1.0):
        self.f_edge = f_edge          # TFLOPS edge
        self.f_server = f_server      # TFLOPS server
        self.bandwidth = bandwidth    # MB/s (vì output đang tính MB)
        self.alpha = alpha            # energy coefficient
        self.V = V                    # Lyapunov weight
