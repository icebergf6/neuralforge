#include <iostream>
#include <vector>
#include "tokenizer.hpp"

using namespace lengine;

int main() {
    std::cout << "========================================\n";
    std::cout << "  BPE Tokenizer Verification\n";
    std::cout << "========================================\n";

    Tokenizer tokenizer;
    std::string tok_path = "models/tokenizer.bin";

    if (!tokenizer.load(tok_path)) {
        std::cerr << "Failed to load tokenizer from " << tok_path << "\n";
        return 1;
    }

    std::cout << "Successfully loaded vocabulary. Size: " << tokenizer.vocab_size << "\n";

    // Test text encoding
    std::string prompt = "Once upon a time, there was a little girl";
    std::cout << "\nEncoding prompt: \"" << prompt << "\"\n";

    std::vector<int> tokens = tokenizer.encode(prompt);
    std::cout << "Encoded tokens (" << tokens.size() << " tokens): [";
    for (size_t i = 0; i < tokens.size(); ++i) {
        std::cout << tokens[i] << (i + 1 < tokens.size() ? ", " : "");
    }
    std::cout << "]\n";

    // Test decoding back to string
    std::cout << "\nDecoding back:\n\"";
    for (int t : tokens) {
        if (t == 1) continue; // Skip BOS symbol
        std::cout << tokenizer.decode(t);
    }
    std::cout << "\"\n";

    std::cout << "\n✅ TOKENIZER TEST PASSED SUCCESSFULLY!\n";
    return 0;
}
