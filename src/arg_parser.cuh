#pragma once

#include <algorithm>
#include <string>
#include <vector>

class ArgParser {
public:
  ArgParser(int argc, char **argv) {
    for (int i = 1; i < argc; ++i) {
      this->args_.push_back(std::string(argv[i]));
    }
  }

  bool cmd_option_exists(const std::string &option) const {
    return std::find(this->args_.begin(), this->args_.end(), option) !=
           this->args_.end();
  }

  template <typename T>
  T get_cmd_option(const std::string &option, T default_value) const {
    auto it = std::find(args_.begin(), args_.end(), option);
    if (it != args_.end() && ++it != args_.end()) {
      try {
        if constexpr (std::is_same_v<T, int>) {
          return std::stoi(*it);
        } else if constexpr (std::is_same_v<T, float>) {
          return std::stof(*it);
        } else if constexpr (std::is_same_v<T, std::string>) {
          return *it;
        }
      } catch (const std::exception &e) {
        fprintf(stderr, "Error: Invalid value for option %s\n", option.c_str());
        exit(EXIT_FAILURE);
      }
    }
    return default_value;
  }

private:
  std::vector<std::string> args_;
};