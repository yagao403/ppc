/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

using namespace std;

constexpr int vector_size = 4;

typedef double double4_t
    __attribute__((vector_size(vector_size * sizeof(double))));

static inline double sum4_t(double4_t x) { return x[0] + x[1] + x[2] + x[3]; }

constexpr double4_t d40{0.0, 0.0, 0.0, 0.0};

void correlate(int ny, int nx, const float *data, float *result) {
  // ceiling of number of vectors per row/col
  int n_vec_per_row = 1 + ((nx - 1) / vector_size);
  int n_vec_per_col = 1 + ((ny - 1) / vector_size);

  int ncd = n_vec_per_col * vector_size;
  vector<double4_t> matrix(ncd * n_vec_per_row, d40);

// normalize input
  vector<double> mean(ny, 0.0);
  vector<double> std(ny, 0.0);

#pragma omp parallel for schedule(static, 1)
  for (int row = 0; row < ny; row++) {
    for (int col = 0; col < nx; col++) {
      mean[row] += data[row * nx + col];
    }
    mean[row] /= nx;
  }

#pragma omp parallel for schedule(static, 1)
  for (int row = 0; row < ny; row++) {
    double square_sum = 0.0;
    for (int col = 0; col < nx; col++) {
      square_sum += pow(data[row * nx + col] - mean[row], 2);
    }
    std[row] = sqrt(square_sum);
  }

// fill in the matrix
#pragma omp parallel for schedule(static, 1)
  for (int row = 0; row < ny; row++) {
    for (int idx_row_vec = 0; idx_row_vec < n_vec_per_row; idx_row_vec++) {
      for (int idx_vec_element = 0; idx_vec_element < vector_size; idx_vec_element++) {
        int col = idx_row_vec * vector_size + idx_vec_element;
        matrix[n_vec_per_row * row + idx_row_vec][idx_vec_element] =
            col < nx ? (data[row * nx + col] - mean[row]) / std[row] : 0.0;
      }
    }
  }

// calculate matrix multiplication X*XT
#pragma omp parallel
#pragma omp for schedule(dynamic, 1) nowait
  for (int row_vec = 0; row_vec < n_vec_per_col; row_vec++) {
    for (int col_vec = row_vec; col_vec < n_vec_per_col; col_vec++) {
      double4_t square_matrix[vector_size][vector_size];
      vector<double4_t> x(vector_size);
      vector<double4_t> y(vector_size);

      // fill up square matrix with zero
      for (int i = 0; i < vector_size; i++) {
        for (int j = 0; j < vector_size; j++) {
          square_matrix[i][j] = d40;
        }
      }

        // matrix multiplication
      for (int idx_row_vec = 0; idx_row_vec < n_vec_per_row; idx_row_vec++) {
        for (int idx_vec_element = 0; idx_vec_element < vector_size; idx_vec_element++) {
          x[idx_vec_element] = matrix[n_vec_per_row * (row_vec * vector_size + idx_vec_element) + idx_row_vec];
          y[idx_vec_element] = matrix[n_vec_per_row * (col_vec * vector_size + idx_vec_element) + idx_row_vec];
        }
        for (int i = 0; i < vector_size; i++) {
          for (int j = 0; j < vector_size; j++) {
            square_matrix[i][j] += x[i] * y[j];
          }
        }
      }

      // fill up result
      for (int i = 0; i < vector_size; i++) {
        for (int j = 0; j < vector_size; j++) {
          int row = row_vec * vector_size + i;
          int col = col_vec * vector_size + j;
          if (row < col && row < ny && col < ny) {
            result[col + row * ny] = static_cast<float>(sum4_t(square_matrix[i][j]));
          }
        }
      }
    }
  }

// diagnal line
#pragma omp parallel for schedule(static, 1)
  for (int row = 0; row < ny; row++) {
    result[row * ny + row] = 1.0;
  }
}