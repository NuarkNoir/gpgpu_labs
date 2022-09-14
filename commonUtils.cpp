#ifndef CUTILS
#define CUTILS

#include <cmath>
#include <random>

float* emptyVec(size_t size) {
    return new float[size];
}

float* generateVec(const size_t size, const float min, const float max) {
    float* vec = new float[size];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(min, max);
    for (size_t i = 0; i < size; i++) {
        vec[i] = dis(gen);
    }
    return vec;
}

float* sumVecs(const float* vec1, const float* vec2, const size_t size) {
    float* sum = new float[size];
    for (size_t i = 0; i < size; i++) {
        sum[i] = vec1[i] + vec2[i];
    }
    return sum;
}

float** generateMat(const size_t rows, const size_t cols, const float min, const float max) {
    float** mat = new float*[rows];
    for (size_t i = 0; i < rows; i++) {
        mat[i] = generateVec(cols, min, max);
    }
    return mat;
}

float** generateSqMat(const size_t size, const float min, const float max) {
    return generateMat(size, size, min, max);
}

float** sumMats(const float** mat1, const float** mat2, const size_t rows, const size_t cols) {
    float** sum = new float*[rows];
    for (size_t i = 0; i < rows; i++) {
        sum[i] = sumVecs(mat1[i], mat2[i], cols);
    }
    return sum;
}

float** sumSqMats(const float** mat1, const float** mat2, const size_t size) {
    return sumMats(mat1, mat2, size, size);
}

bool compareVecs(const float* vec1, const float* vec2, const size_t size) {
    for (size_t i = 0; i < size; i++) {
        if (std::fabs(vec1[i] - vec2[i]) >= 0.0001f) {
            fprintf(stderr, "vec1[%zu] = %f, vec2[%zu] = %f (adiff = %f)\n", i, vec1[i], i, vec2[i], std::fabs(vec1[i] - vec2[i]));
            return false;
        }
    }
    return true;
}

bool compareMats(const float** mat1, const float** mat2, const size_t rows, const size_t cols) {
    for (size_t i = 0; i < rows; i++) {
        if (!compareVecs(mat1[i], mat2[i], cols)) {
            return false;
        }
    }
    return true;
}

bool compareSqMats(const float** mat1, const float** mat2, const size_t size) {
    return compareMats(mat1, mat2, size, size);
}

void printVec(const float* vec, const size_t size, FILE* stream = stdout) {
    for (size_t i = 0; i < size; i++) {
        fprintf(stream, "%f ", vec[i]);
    }
    fprintf(stream, "\n");
}

void printMat(const float** mat, const size_t rows, const size_t cols, FILE* stream = stdout) {
    for (size_t i = 0; i < rows; i++) {
        printVec(mat[i], cols, stream);
    }
}

void printSqMat(const float** mat, const size_t size, FILE* stream = stdout) {
    printMat(mat, size, size, stream);
}

#endif // CUTILS