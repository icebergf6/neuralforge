#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include "ops.hpp"

using namespace lengine;

void test_rmsnorm() {
    std::cout << "[TEST] Running RMSNorm test... ";
    const int size = 4;
    float x[size] = {1.0f, 2.0f, 3.0f, 4.0f};
    float weight[size] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[size];

    Ops::rmsnorm(out, x, weight, size, 1e-5f);

    // Hitung mean square: (1 + 4 + 9 + 16) / 4 = 30 / 4 = 7.5
    // rms = sqrt(7.5) ~= 2.7386
    float expected_0 = 1.0f / std::sqrt(7.5f);
    assert(std::abs(out[0] - expected_0) < 1e-3);
    std::cout << "PASSED (out[0] = " << out[0] << ")\n";
}

void test_matmul() {
    std::cout << "[TEST] Running MatMul test... ";
    // W = [[1, 2], [3, 4]], x = [2, 3] -> out = [1*2 + 2*3, 3*2 + 4*3] = [8, 18]
    float W[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float x[2] = {2.0f, 3.0f};
    float out[2];

    Ops::matmul(out, x, W, 2, 2);

    assert(std::abs(out[0] - 8.0f) < 1e-5);
    assert(std::abs(out[1] - 18.0f) < 1e-5);
    std::cout << "PASSED (out = [" << out[0] << ", " << out[1] << "])\n";
}

void test_softmax() {
    std::cout << "[TEST] Running Softmax test... ";
    float x[3] = {1.0f, 2.0f, 3.0f};
    Ops::softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    assert(std::abs(sum - 1.0f) < 1e-5);
    assert(x[2] > x[1] && x[1] > x[0]);
    std::cout << "PASSED (sum = " << sum << ", max_prob = " << x[2] << ")\n";
}

void test_swiglu() {
    std::cout << "[TEST] Running SwiGLU test... ";
    float gate[2] = {0.0f, 2.0f};
    float up[2] = {5.0f, 3.0f};
    float out[2];

    Ops::swiglu(out, gate, up, 2);

    // silu(0) = 0 / 2 = 0 -> out[0] = 0 * 5 = 0
    assert(std::abs(out[0] - 0.0f) < 1e-5);
    std::cout << "PASSED\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Core Math Operations Unit Tests\n";
    std::cout << "========================================\n";

    test_rmsnorm();
    test_matmul();
    test_softmax();
    test_swiglu();

    std::cout << "\n✅ ALL MATH KERNEL UNIT TESTS PASSED SUCCESSFULLY!\n";
    return 0;
}
