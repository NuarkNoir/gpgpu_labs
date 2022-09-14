#include <iostream>
#include <fstream>
#include "../commonUtils.cpp"

int main(int argc, char** argv) {
    size_t size = 0;
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <size>" << std::endl;
        return 1;
    }

    size = std::atoll(argv[1]);
    
    volatile auto a = generateVec(size, -100, 100);
    volatile auto b = generateVec(size, -100, 100);
    volatile auto sumE = sumVecs(a, b, size);

    (void)sumE;

    return 0;
}