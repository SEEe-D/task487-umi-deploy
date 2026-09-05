#include "stall_release.hpp"

#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

int main(int argc, char **argv) {
  using gripper_control::requests_stall_release;
  constexpr double deadband = 0.017453292519943295;
  auto require = [](bool ok, const char *message) {
    if (!ok) throw std::runtime_error(message);
  };
  // Recorded CNC event: object prevents reaching the extra-close hold. The
  // old comparison with hold=-0.1363 released on this still-closing request.
  require(!requests_stall_release(-0.1687, -0.1712, deadband), "closing contact must remain latched");
  require(!requests_stall_release(-0.1712, -0.1712, deadband), "hold must remain latched");
  require(!requests_stall_release(-0.1800, -0.1712, deadband), "sub-deadband noise must not release");
  require(requests_stall_release(-0.2000, -0.1712, deadband), "deliberate opening must release");
  // If compression really moves the finger, opening is relative to its new
  // position; a frozen contact reference would incorrectly inhibit release.
  require(requests_stall_release(-0.1600, -0.1363, deadband), "release after compression must work");
  require(requests_stall_release(-0.035, 0.0, deadband), "closed stop must reopen");
  require(!requests_stall_release(std::numeric_limits<double>::quiet_NaN(), -0.17, deadband), "invalid command must not release");
  require(!requests_stall_release(-0.2, std::numeric_limits<double>::quiet_NaN(), deadband), "invalid feedback must not release");

  int count = 0;
  if (argc == 2) {
    std::ifstream input(argv[1]);
    require(input.good(), "replay file missing");
    double requested, measured;
    int expected;
    while (input >> requested >> measured >> expected) {
      require(requests_stall_release(requested, measured, deadband) == bool(expected),
              "recorded contact replay mismatch");
      ++count;
    }
    require(input.eof(), "malformed replay row");
    require(count > 0, "empty replay");
  }
  std::cout << "8 stall release cases passed; " << count << " recorded cases passed\n";
}
