// File: main.cpp
#include <csignal>
#include <iostream>
#include <jsoncpp/json/json.h>
#include <fstream>
#include <atomic>
#include <thread>
#include <chrono>
#include <filesystem>  // C++17


std::atomic<bool> should_reload(false);

void signal_handler(int signum) {
  if (signum == SIGUSR1) {
    should_reload = true;
  }
}

void write_reload_flag() {
  std::ofstream flag("reload.flag");
  flag << "1";
  flag.close();
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

int main() {
  std::signal(SIGUSR1, signal_handler);
  const std::string config_path = "config.json";

  while (true) {
    Config cfg = load_config(config_path);
    std::cout << "[Rank " << cfg.rank << "] Config loaded. Waiting for signal to reload...\n";
    while (!should_reload) {
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    should_reload = false;
    std::cout << "[Rank " << cfg.rank << "] Received reload signal.\n";
    write_reload_flag();
  }

  return 0;
}
