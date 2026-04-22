#include <iostream>
#include <fstream>
#include <chrono>
#include <string>
#include <vector>
#include <cstring>
#include <mpi.h>

#include "matrix_math.h"

using namespace std;

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    setlocale(LC_ALL, "Russian");
    
    string fileA, fileB, fileOut;
    const string defA = "dataA.txt";
    const string defB = "dataB.txt";
    const string defOut = "output.txt";
    
    if (rank == 0) {
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
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        testA.close();
        
        ifstream testB(fileB);
        if (!testB.is_open()) {
            cerr << "Error: cannot open file '" << fileB << "'\n";
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        testB.close();
    }
    
    const int nameLen = 256;
    char* fileABuffer = new char[nameLen];
    char* fileBBuffer = new char[nameLen];
    char* fileOutBuffer = new char[nameLen];
    
    if (rank == 0) {
        strcpy_s(fileABuffer, nameLen, fileA.c_str());
        strcpy_s(fileBBuffer, nameLen, fileB.c_str());
        strcpy_s(fileOutBuffer, nameLen, fileOut.c_str());
    }
    
    MPI_Bcast(fileABuffer, nameLen, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(fileBBuffer, nameLen, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Bcast(fileOutBuffer, nameLen, MPI_CHAR, 0, MPI_COMM_WORLD);
    
    string fileA_mpi(fileABuffer);
    string fileB_mpi(fileBBuffer);
    string fileOut_mpi(fileOutBuffer);
    
    delete[] fileABuffer;
    delete[] fileBBuffer;
    delete[] fileOutBuffer;
    
    try {
        Matr A, B;
        size_t N = 0;
        
        if (rank == 0) {
            cout << "\nReading first matrix from " << fileA_mpi << "\n";
            A = loadMatrix(fileA_mpi);
            
            cout << "Reading second matrix from " << fileB_mpi << "\n";
            B = loadMatrix(fileB_mpi);
            
            if (A.getRows() != A.getCols() || B.getRows() != B.getCols()) {
                cerr << "Error: both matrices must be square.\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            if (A.getRows() != B.getRows()) {
                cerr << "Error: dimension mismatch. A is " << A.getRows()
                     << "x" << A.getCols() << ", B is " << B.getRows()
                     << "x" << B.getCols() << "\n";
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            
            N = A.getRows();
            cout << "Matrix size: " << N << " x " << N << "\n";
            cout << "Number of MPI processes: " << size << "\n";
            cout << "Approximate operations count: " << N * N * N << "\n";
        }
        
        MPI_Bcast(&N, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        
        int rows_per_proc = N / size;
        int remainder = N % size;
        int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);
        
        vector<float> A_local(my_rows * N);
        vector<float> B_full(N * N);
        
        if (rank == 0) {
            vector<int> sendcounts(size);
            vector<int> displs(size);
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows = rows_per_proc + (i < remainder ? 1 : 0);
                sendcounts[i] = rows * N;
                displs[i] = offset;
                offset += sendcounts[i];
            }
            
            vector<float> A_full(N * N);
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    A_full[i * N + j] = A(i, j);
                }
            }
            
            MPI_Scatterv(A_full.data(), sendcounts.data(), displs.data(), MPI_FLOAT,
                         A_local.data(), my_rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
            
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    B_full[i * N + j] = B(i, j);
                }
            }
        } else {
            MPI_Scatterv(nullptr, nullptr, nullptr, MPI_FLOAT,
                         A_local.data(), my_rows * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        }
        
        MPI_Bcast(B_full.data(), N * N, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        vector<float> C_local(my_rows * N, 0.0f);
        
        MPI_Barrier(MPI_COMM_WORLD);
        auto start = chrono::high_resolution_clock::now();
        
        for (int i = 0; i < my_rows; i++) {
            for (size_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < N; k++) {
                    sum += A_local[i * N + k] * B_full[k * N + j];
                }
                C_local[i * N + j] = sum;
            }
        }
        
        auto end = chrono::high_resolution_clock::now();
        
        if (rank == 0) {
            auto elapsedMs = chrono::duration_cast<chrono::milliseconds>(end - start);
            auto elapsedUs = chrono::duration_cast<chrono::microseconds>(end - start);
            cout << "Multiplication took: " << elapsedMs.count() << " ms ("
                 << elapsedUs.count() << " µs)\n";
        }
        
        if (rank == 0) {
            vector<int> recvcounts(size);
            vector<int> displs(size);
            int offset = 0;
            for (int i = 0; i < size; i++) {
                int rows = rows_per_proc + (i < remainder ? 1 : 0);
                recvcounts[i] = rows * N;
                displs[i] = offset;
                offset += recvcounts[i];
            }
            
            vector<float> C_full(N * N);
            MPI_Gatherv(C_local.data(), my_rows * N, MPI_FLOAT,
                        C_full.data(), recvcounts.data(), displs.data(), MPI_FLOAT,
                        0, MPI_COMM_WORLD);
            
            Matr C(N, N);
            for (size_t i = 0; i < N; i++) {
                for (size_t j = 0; j < N; j++) {
                    C(i, j) = C_full[i * N + j];
                }
            }
            
            cout << "Saving result to " << fileOut_mpi << " ...\n";
            saveMatrix(fileOut_mpi, C);
            cout << "Done.\n";
        } else {
            MPI_Gatherv(C_local.data(), my_rows * N, MPI_FLOAT,
                        nullptr, nullptr, nullptr, MPI_FLOAT,
                        0, MPI_COMM_WORLD);
        }
        
    } catch (const exception& ex) {
        if (rank == 0) {
            cerr << "Exception: " << ex.what() << "\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    
    MPI_Finalize();
    return 0;
}