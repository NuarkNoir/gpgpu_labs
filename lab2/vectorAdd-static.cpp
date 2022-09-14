#include <iostream>
#include <fstream>
#include "../commonUtils.cpp"

int main() {
    size_t size = 0;

    std::ifstream data("data.txt", std::ios::in);
    data >> size;
    
    auto a = emptyVec(size);
    auto b = emptyVec(size);
    auto sumE = emptyVec(size);

    for (size_t i = 0; i < size; i++) {
        data >> a[i] >> b[i] >> sumE[i];
    }
    
    auto sumR = sumVecs(a, b, size);

    bool veq = compareVecs(sumE, sumR, size);
    
    if (veq) {
        std::cout << "Success!" << std::endl;
        return 0;
    }
    
    std::cout << "Failure!" << std::endl;
    return 1;
}