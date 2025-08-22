#include <algorithm>

#include "arg_parser.cuh"

ArgParser::ArgParser(int argc, char **argv) {
  for (int i = 1; i < argc; ++i) {
    this->args_.push_back(std::string(argv[i]));
  }
}

bool ArgParser::cmd_option_exists(const std::string &option) const {
  return std::find(this->args_.begin(), this->args_.end(), option) !=
         this->args_.end();
}