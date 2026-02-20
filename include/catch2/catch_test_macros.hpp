#pragma once

#include <exception>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace catch2_compat {

using TestFn = std::function<void()>;

struct TestCase {
    std::string name;
    TestFn fn;
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

struct Registrar {
    Registrar(std::string name, TestFn fn) { registry().push_back({std::move(name), std::move(fn)}); }
};

}  // namespace catch2_compat

#define CATCH2_COMPAT_JOIN_IMPL(a, b) a##b
#define CATCH2_COMPAT_JOIN(a, b) CATCH2_COMPAT_JOIN_IMPL(a, b)

#define TEST_CASE(name, tags)                                                                                           \
    static void CATCH2_COMPAT_JOIN(test_fn_, __LINE__)();                                                              \
    static ::catch2_compat::Registrar CATCH2_COMPAT_JOIN(test_reg_, __LINE__)(name, CATCH2_COMPAT_JOIN(test_fn_, __LINE__)); \
    static void CATCH2_COMPAT_JOIN(test_fn_, __LINE__)()

#define REQUIRE(expr)                                                                                                   \
    do {                                                                                                                \
        if (!(expr)) {                                                                                                  \
            throw std::runtime_error(std::string("REQUIRE failed: ") + #expr);                                       \
        }                                                                                                               \
    } while (false)

#ifdef CATCH_CONFIG_MAIN
int main() {
    int failed = 0;
    for (const auto& tc : ::catch2_compat::registry()) {
        try {
            tc.fn();
            std::cout << "[PASS] " << tc.name << '\n';
        } catch (const std::exception& ex) {
            ++failed;
            std::cerr << "[FAIL] " << tc.name << ": " << ex.what() << '\n';
        }
    }
    return failed == 0 ? 0 : 1;
}
#endif
