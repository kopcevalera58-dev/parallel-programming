#include "matrix_math.h"
#include <fstream>
#include <stdexcept>

using namespace std;

Matr::Matr(size_t r, size_t c) : data_(r * c, 0.0f), rows_(r), cols_(c) {}

size_t Matr::getRows() const { return rows_; }
size_t Matr::getCols() const { return cols_; }

float Matr::get(size_t i, size_t j) const {
    return data_[i * cols_ + j];
}

void Matr::set(size_t i, size_t j, float val) {
    data_[i * cols_ + j] = val;
}

float Matr::operator()(size_t i, size_t j) const {
    return data_[i * cols_ + j];
}

float& Matr::operator()(size_t i, size_t j) {
    return data_[i * cols_ + j];
}

Matr operator*(const Matr& lhs, const Matr& rhs) {
    if (lhs.getCols() != rhs.getRows()) {
        throw std::invalid_argument("Matrix dimensions mismatch");
    }
    
    Matr result(lhs.getRows(), rhs.getCols());
    
    for (size_t i = 0; i < lhs.getRows(); ++i) {
        for (size_t j = 0; j < rhs.getCols(); ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < lhs.getCols(); ++k) {
                sum += lhs(i, k) * rhs(k, j);
            }
            result(i, j) = sum;
        }
    }
    return result;
}

Matr loadMatrix(const string& fname) {
    ifstream file(fname);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + fname);
    }
    size_t n;
    file >> n;
    if (!file || n == 0) {
        throw runtime_error("Invalid matrix size");
    }
    Matr res(n, n);
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            float val;
            if (!(file >> val)) {
                throw runtime_error("Not enough data");
            }
            res(i, j) = val;
        }
    }
    return res;
}

void saveMatrix(const string& fname, const Matr& m) {
    ofstream file(fname);
    if (!file.is_open()) {
        throw runtime_error("Cannot create file: " + fname);
    }
    file << m.getRows() << '\n';
    for (size_t i = 0; i < m.getRows(); ++i) {
        for (size_t j = 0; j < m.getCols(); ++j) {
            file << m(i, j) << ' ';
        }
        file << '\n';
    }
}