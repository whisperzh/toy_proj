// File: main.cpp
#include <nccl.h>
#include <cuda_runtime.h>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <iostream>
#include <thread>
#include <chrono>
#include <csignal>
#include <atomic>

#define CHECK(cmd) do { \
  cudaError_t e = cmd; \
  if (e != cudaSuccess) { \
    std::cerr << "Failed: " << #cmd << " (error: " << cudaGetErrorString(e) << ")\n"; \
    exit(EXIT_FAILURE); \
  } \
} while(0)

#define NCCLCHECK(cmd) do { \
  ncclResult_t r = cmd; \
  if (r != ncclSuccess) { \
    std::cerr << "NCCL Failure: " << ncclGetErrorString(r) << "\n"; \
    exit(EXIT_FAILURE); \
  } \
} while(0)

std::atomic<bool> should_reload(false);

void signal_handler(int signum) {
  if (signum == SIGUSR1) {
    should_reload = true;
  }
}

struct Config {
  int rank;
  int world_size;
  std::string master_ip;
};

Config load_config(const std::string& path) {
  std::ifstream file(path);
  Json::Value root;
  file >> root;

  Config cfg;
  cfg.rank = root["rank"].asInt();
  cfg.world_size = root["world_size"].asInt();
  cfg.master_ip = root["master_ip"].asString();
  return cfg;
}

void run_nccl(Config cfg) {
  ncclComm_t comm;
  ncclUniqueId id;
  cudaStream_t stream;

  int size = 1024 * 1024 * 100; // 100MB
  float* send_buff;
  float* recv_buff;
  CHECK(cudaMalloc(&send_buff, size * sizeof(float)));
  CHECK(cudaMalloc(&recv_buff, size * sizeof(float)));
  CHECK(cudaStreamCreate(&stream));

  if (cfg.rank == 0) {
    NCCLCHECK(ncclGetUniqueId(&id));
    std::ofstream out("nccl_id.bin", std::ios::binary);
    out.write((char*)&id, sizeof(id));
    out.close();
  } else {
    while (!std::ifstream("nccl_id.bin")) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::ifstream in("nccl_id.bin", std::ios::binary);
    in.read((char*)&id, sizeof(id));
    in.close();
  }

  NCCLCHECK(ncclCommInitRank(&comm, cfg.world_size, id, cfg.rank));

  std::cout << "[Rank " << cfg.rank << "] NCCL Init Done\n";

  for (int iter = 0; iter < 5; ++iter) {
    if (should_reload) break;

    auto start = std::chrono::high_resolution_clock::now();
    if (cfg.rank == 0) {
      NCCLCHECK(ncclSend(send_buff, size, ncclFloat, 1, comm, stream));
      NCCLCHECK(ncclRecv(recv_buff, size, ncclFloat, 1, comm, stream));
    } else if (cfg.rank == 1) {
      NCCLCHECK(ncclRecv(recv_buff, size, ncclFloat, 0, comm, stream));
      NCCLCHECK(ncclSend(recv_buff, size, ncclFloat, 0, comm, stream));
    }
    CHECK(cudaStreamSynchronize(stream));
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    std::cout << "[Rank " << cfg.rank << "] Iter " << iter << " RTT: " << ms << " ms\n";
  }

  NCCLCHECK(ncclCommDestroy(comm));
  CHECK(cudaFree(send_buff));
  CHECK(cudaFree(recv_buff));
  CHECK(cudaStreamDestroy(stream));
  std::cout << "[Rank " << cfg.rank << "] Shutdown\n";
}

int main() {
  std::signal(SIGUSR1, signal_handler);
  const std::string config_path = "config.json";

  while (true) {
    Config cfg = load_config(config_path);
    should_reload = false;
    run_nccl(cfg);
    if (!should_reload) break;
    std::cout << "[Rank " << cfg.rank << "] Reloading NCCL group...\n";
  }
  return 0;
}
