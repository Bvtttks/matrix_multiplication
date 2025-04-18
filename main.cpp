#include <vector>
#include <stdexcept>
#include <iostream>
#include <cstdint>

template <typename T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix) {
    if (matrix.empty()) return {};

    size_t original_rows = matrix.size();
    size_t original_cols = matrix[0].size();

    std::vector<std::vector<T>> result(original_cols, std::vector<T>(original_rows));

    for (size_t i = 0; i < original_rows; ++i) {
        for (size_t j = 0; j < original_cols; ++j) {
            result[j][i] = matrix[i][j];
        }
    }

    return result;
}

std::vector<std::vector<uint32_t>> simple_matrix_mul(
    const std::vector<std::vector<uint16_t>>& matr1,
    const std::vector<std::vector<uint16_t>>& matr2
) {
    if (matr1.empty() || matr2.empty()) {
        throw std::invalid_argument("One or both matrices are empty.");
    }

    size_t matr1_rows = matr1.size();
    size_t matr1_cols = matr1[0].size();
    size_t matr2_rows = matr2.size();
    size_t matr2_cols = matr2[0].size();

    if (matr1_cols != matr2_rows) {
        throw std::invalid_argument("Matrix sizes are incompatible for multiplication.");
    }

    std::vector<std::vector<uint32_t>> result(matr1_rows, std::vector<uint32_t>(matr2_cols, 0));

    for (size_t i = 0; i < matr1_rows; ++i) {
        for (size_t j = 0; j < matr2_cols; ++j) {
            for (size_t k = 0; k < matr1_cols; ++k) {
                result[i][j] += static_cast<uint32_t>(matr1[i][k]) * static_cast<uint32_t>(matr2[k][j]);
            }
        }
    }

    return result;
}

std::vector<std::vector<uint32_t>> primitive(
    const std::vector<std::vector<uint16_t>>& matr1_block,
    const std::vector<std::vector<uint16_t>>& matr2_block)
{
    if (matr1_block.size() != 8 || matr1_block[0].size() != 4 ||
        matr2_block.size() != 4 || matr2_block[0].size() != 16) {
        throw std::invalid_argument("The block sizes should be 8x4 and 4x16");
    }

    std::vector<std::vector<uint32_t>> result_block(8, std::vector<uint32_t>(16, 0));

    for (int row = 0; row < 8; ++row) {
        for (int col = 0; col < 16; ++col) {
            for (int k = 0; k < 4; ++k) {
                result_block[row][col] += static_cast<uint32_t>(matr1_block[row][k]) * matr2_block[k][col];
            }
        }
    }

    return result_block;
}

std::vector<std::vector<uint32_t>> multiplyLargeMatrices(
    const std::vector<std::vector<uint16_t>>& matr1,
    const std::vector<std::vector<uint16_t>>& matr2)
{
    size_t m1_height = matr1.size(); // количество строк в первой матрице
    size_t m1_width  = matr1[0].size(); // количество столбцов в первой матрице
    size_t m2_width  = matr2[0].size();

    if (m1_width != matr2.size()) {
        throw std::invalid_argument("Matrix sizes are incompatible for multiplication.");
    }

    std::vector<std::vector<uint32_t>> result_matrix(m1_height, std::vector<uint32_t>(m2_width, 0));

    for (size_t row_block = 0; row_block < m1_height; row_block += 8) {
        for (size_t col_block = 0; col_block < m2_width; col_block += 16) {
            for (size_t shared_block = 0; shared_block < m1_width; shared_block += 4) {

                // выделяем блок 8x4 из первой матрицы
                std::vector<std::vector<uint16_t>> block_from_matr1(8, std::vector<uint16_t>(4));
                for (size_t i = 0; i < 8; ++i)
                    for (size_t j = 0; j < 4; ++j)
                        block_from_matr1[i][j] = matr1[row_block + i][shared_block + j];

                // выделяем блок 4x16 из второй матрицы
                std::vector<std::vector<uint16_t>> block_from_matr2(4, std::vector<uint16_t>(16));
                for (size_t i = 0; i < 4; ++i)
                    for (size_t j = 0; j < 16; ++j)
                        block_from_matr2[i][j] = matr2[shared_block + i][col_block + j];

                auto partial_result = primitive(block_from_matr1, block_from_matr2);

                // добавляем результат в нужное место
                for (size_t i = 0; i < 8; ++i)
                    for (size_t j = 0; j < 16; ++j)
                        result_matrix[row_block + i][col_block + j] += partial_result[i][j];
            }
        }
    }

    return result_matrix;
}

int main() {
    const int matr1_rows = 8;
    const int matr1_cols = 4;
    const int matr2_cols = 16;

    std::vector<std::vector<uint16_t>> matr1(matr1_rows, std::vector<uint16_t>(matr1_cols));
    std::vector<std::vector<uint16_t>> matr2(matr1_cols, std::vector<uint16_t>(matr2_cols)); // matr2_rows = matr1_cols

    uint16_t val = 1;
    for (int i = 0; i < matr1_rows; ++i)
        for (int j = 0; j < matr1_cols; ++j)
            matr1[i][j] = val++;

    val = 1;
    for (int i = 0; i < matr1_cols; ++i)
        for (int j = 0; j < matr2_cols; ++j)
            matr2[i][j] = val++;

    std::vector<std::vector<uint32_t>> reference = simple_matrix_mul(matr1, matr2);
    auto result = multiplyLargeMatrices(matr1, matr2);

    bool match = true;
    for (int i = 0; i < matr1_rows; ++i)
        for (int j = 0; j < matr2_cols; ++j)
            if (result[i][j] != reference[i][j]) {
                std::cout << "Mismatch at (" << i << ", " << j << "): "
                          << result[i][j] << " != " << reference[i][j] << "\n";
                match = false;
            }

    if (match) {
        std::cout << "Результат совпал с эталоном." << std::endl;
    } else {
        std::cout << "Результат НЕ совпал с эталоном, реализация ошибочна." << std::endl;
    }

    return 0;
}
