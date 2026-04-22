#ifndef MATRIX_MATH_H
#define MATRIX_MATH_H

#include <vector>
#include <string>
#include <cstddef>

class Matr {
private:
    std::vector<float> data_;
    size_t rows_;
    size_t cols_;
    
public:
    Matr(size_t r = 0, size_t c = 0);
    
    size_t getRows() const;
    size_t getCols() const;
    
    float get(size_t i, size_t j) const;
    void set(size_t i, size_t j, float val);
    
    float operator()(size_t i, size_t j) const;
    float& operator()(size_t i, size_t j);
};

Matr operator*(const Matr& lhs, const Matr& rhs);
Matr loadMatrix(const std::string& fname);
void saveMatrix(const std::string& fname, const Matr& m);

#endif