#include <iostream>
#include <string>
#include "engine.hpp"

using namespace lengine;

int main(int argc, char* argv[]) {
    std::cout << "=========================================================\n";
    std::cout << "  LeagueOfLLMs: Custom C++ LLM Inference Engine (from scratch)\n";
    std::cout << "=========================================================\n";

    std::string model_path = "models/stories15M.bin";
    std::string tokenizer_path = "models/tokenizer.bin";
    std::string prompt = "One day, a curious little robot named Bolt";

    if (argc > 1) {
        prompt = argv[1];
    }

    Engine engine;
    if (!engine.init(model_path, tokenizer_path)) {
        std::cerr << "Initialization failed!\n";
        return 1;
    }

    // Run text generation
    engine.generate(prompt, 64, 0.8f, 0.9f);

    return 0;
}
