#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <clocale>
#include <omp.h>           

#include "matrix_math.h"

int main() {
    setlocale(LC_ALL, ".UTF-8");   

    std::string fileA, fileB, fileOut;
    const std::string defA = "dataA.txt";
    const std::string defB = "dataB.txt";
    const std::string defOut = "output.txt";

    std::cout << "Enter matrix A file name [" << defA << "]: ";
    std::getline(std::cin, fileA);
    if (fileA.empty()) fileA = defA;

    std::cout << "Enter matrix B file name [" << defB << "]: ";
    std::getline(std::cin, fileB);
    if (fileB.empty()) fileB = defB;

    std::cout << "Enter output file name [" << defOut << "]: ";
    std::getline(std::cin, fileOut);
    if (fileOut.empty()) fileOut = defOut;

    int num_threads;
    std::cout << "Enter number of threads: ";
    std::cin >> num_threads;
    omp_set_num_threads(num_threads);

    std::ifstream testA(fileA);
    if (!testA.is_open()) {
        std::cerr << "Error: cannot open file '" << fileA << "'\n";
        return 1;
    }
    testA.close();

    std::ifstream testB(fileB);
    if (!testB.is_open()) {
        std::cerr << "Error: cannot open file '" << fileB << "'\n";
        return 1;
    }
    testB.close();

    try {
        std::cout << "\nReading first matrix from " << fileA << "\n";
        Matr A = loadMatrix(fileA);

        std::cout << "Reading second matrix from " << fileB << "\n";
        Matr B = loadMatrix(fileB);

        if (A.getRows() != A.getCols() || B.getRows() != B.getCols()) {
            std::cerr << "Error: both matrices must be square.\n";
            return 1;
        }
        if (A.getRows() != B.getRows()) {
            std::cerr << "Error: dimension mismatch. A is " << A.getRows()
                      << "x" << A.getCols() << ", B is " << B.getRows()
                      << "x" << B.getCols() << "\n";
            return 1;
        }

        size_t n = A.getRows();
        std::cout << "Matrix size: " << n << " x " << n << "\n";
        size_t totalOps = n * n * n;
        std::cout << "Approximate operations count: " << totalOps << "+-1\n";

        auto start = std::chrono::high_resolution_clock::now();
        Matr C = mul(A, B);
        auto end = std::chrono::high_resolution_clock::now();

        auto elapsedMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
        auto elapsedUs = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Multiplication took: " << elapsedMs.count() << " ms ("
                  << elapsedUs.count() << " µs)\n";

        std::cout << "Saving result to " << fileOut << " ...\n";
        saveMatrix(fileOut, C);

        std::cout << "Done.\n";
    } catch (const std::exception& ex) {
        std::cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}