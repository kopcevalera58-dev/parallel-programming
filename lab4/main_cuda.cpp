#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include "matrix_math.h"
#include "matrix_math_cuda.h"

using namespace std;

int main(int argc, char* argv[]) {
    setlocale(LC_ALL, "Russian");
    
    try {
        if (argc > 1 && string(argv[1]) == "--experiment") {
            runCudaExperiments();
            return 0;
        }
        
        string fileA, fileB, fileOut;
        const string defA = "dataA.txt";
        const string defB = "dataB.txt";
        const string defOut = "output.txt";

        cout << "Enter matrix A file name [" << defA << "]: ";
        getline(cin, fileA);
        if (fileA.empty()) fileA = defA;

        cout << "Enter matrix B file name [" << defB << "]: ";
        getline(cin, fileB);
        if (fileB.empty()) fileB = defB;

        cout << "Enter output file name [" << defOut << "]: ";
        getline(cin, fileOut);
        if (fileOut.empty()) fileOut = defOut;

        ifstream testA(fileA);
        if (!testA.is_open()) {
            cerr << "Error: cannot open file '" << fileA << "'\n";
            return 1;
        }
        testA.close();

        ifstream testB(fileB);
        if (!testB.is_open()) {
            cerr << "Error: cannot open file '" << fileB << "'\n";
            return 1;
        }
        testB.close();

        cout << "\nReading first matrix from " << fileA << "\n";
        Matr A = loadMatrix(fileA);

        cout << "Reading second matrix from " << fileB << "\n";
        Matr B = loadMatrix(fileB);

        if (A.getRows() != A.getCols() || B.getRows() != B.getCols()) {
            cerr << "Error: both matrices must be square.\n";
            return 1;
        }
        if (A.getRows() != B.getRows()) {
            cerr << "Error: dimension mismatch.\n";
            return 1;
        }

        size_t n = A.getRows();
        cout << "Matrix size: " << n << " x " << n << "\n";
        
        printDeviceInfo();
        
        vector<float> flatA(n * n);
        vector<float> flatB(n * n);
        vector<float> flatC(n * n);
        
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                flatA[i * n + j] = A.get(i, j);
                flatB[i * n + j] = B.get(i, j);
            }
        }
        
        double timeMs;
        cout << "Running CUDA multiplication...\n";
        
        auto start = chrono::high_resolution_clock::now();
        benchmarkCUDA(flatA.data(), flatB.data(), flatC.data(),
                     n, n, n, 16, 16, timeMs);
        auto end = chrono::high_resolution_clock::now();
        
        auto elapsedUs = chrono::duration_cast<chrono::microseconds>(end - start);
        
        cout << "CUDA multiplication took: " << timeMs << " ms (" 
             << timeMs * 1000 << " µs)\n";
        cout << "Total with overhead: " << elapsedUs.count() << " µs\n";
        
        Matr C(n, n);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                C.set(i, j, flatC[i * n + j]);
            }
        }
        
        cout << "Saving result to " << fileOut << " ...\n";
        saveMatrix(fileOut, C);
        cout << "Done.\n";
        
    } catch (const std::exception& ex) {
        cerr << "Exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}