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
// #include <iostream>
#include <numeric>
#include <vector>

using namespace std;

constexpr int vector_size = 4;

typedef double double4_t
    __attribute__((vector_size(vector_size * sizeof(double))));

static inline double sum4_t(double4_t x) { return x[0] + x[1] + x[2] + x[3]; }

constexpr double4_t d40{0.0, 0.0, 0.0, 0.0};

void correlate(int ny, int nx, const float *data, float *result) {

  vector<double> intermediate_result(ny * nx);
  double sum, square_sum, mean, x;

  // ceiling of number of vectors per row
  int n_vec_per_row = 1 + ((nx - 1) / vector_size);
  int pad_size = nx % vector_size;
  // number of vectors
  int n_vec = ny * n_vec_per_row;
  // define the matrix
  vector<double4_t> matrix(n_vec);
  // fill matrix with 0.0
  fill(matrix.begin(), matrix.end(), d40);

  //   cout << "vector per row: " << n_vec_per_row << endl;
  //   cout << "matrix size: " << matrix.size() << endl;

  for (int row = 0; row < ny; row++) {
    sum = 0.0;

    for (int col = 0; col < nx; col++) {
      sum += data[col + row * nx];
    }

    mean = sum / nx;

    square_sum = 0.0;

    for (int col = 0; col < nx; col++) {
      x = data[col + row * nx] - mean;
      intermediate_result[col + row * nx] = x;
      square_sum += x * x;
    }

    square_sum = sqrt(square_sum);

    for (int col = 0; col < nx; col++) {
      intermediate_result[col + row * nx] /= square_sum;
    }
  }

  if (pad_size == 0) {
    // each vector are full of elements
    for (int row = 0; row < ny; row++) {
      // fill the vector matrix
      int col;

      // fill each row (except for the last vector)
      for (int idx_vec = 0; idx_vec < n_vec_per_row; idx_vec++) {
        for (int idx_element = 0; idx_element < vector_size; idx_element++) {
          col = idx_vec * vector_size + idx_element;
          matrix[row * n_vec_per_row + idx_vec][idx_element] =
              intermediate_result[col + row * nx];
        }
      }
    }
  } else {
    // some vectors are partially full
    for (int row = 0; row < ny; row++) {
      // fill the vector matrix
      int col;

      // fill each row (except for the last vector)
      for (int idx_vec = 0; idx_vec < n_vec_per_row - 1; idx_vec++) {
        for (int idx_element = 0; idx_element < vector_size; idx_element++) {
          col = idx_vec * vector_size + idx_element;
          matrix[row * n_vec_per_row + idx_vec][idx_element] =
              intermediate_result[col + row * nx];
        }
      }

      // fill the last vector in a row
      for (int idx_element = 0; idx_element < pad_size; idx_element++) {
        col = vector_size * (n_vec_per_row - 1) + idx_element;
        matrix[row * n_vec_per_row + n_vec_per_row - 1][idx_element] =
            intermediate_result[col + row * nx];
      }
    }
  }
  //   cout<<"Pad: "<<pad_size<<endl;
  //   cout << "Last vector in row 0"
  //        << ": " << matrix[0 * n_vec_per_row + n_vec_per_row - 1][0]
  //        << matrix[0 * n_vec_per_row + n_vec_per_row - 1][1]
  //        << matrix[0 * n_vec_per_row + n_vec_per_row - 1][2]
  //        << matrix[0 * n_vec_per_row + n_vec_per_row - 1][3] << endl;

  //   cout << "Matrix size: " << matrix.size() << endl;
  //   cout << "Matrix sum: " << sum4_t(reduce(matrix.begin(), matrix.end(),
  //   d40))
  //        << endl;

  double4_t corr;
  for (int row_1 = 0; row_1 < ny; row_1++) {
    for (int row_2 = row_1; row_2 < ny; row_2++) {
      corr = d40;
      asm("# loop starts here");
      // calcualte vector-wise multiplication
      for (int vector_idx = 0; vector_idx < n_vec_per_row; vector_idx++) {
        corr += matrix[row_1 * n_vec_per_row + vector_idx] *
                matrix[row_2 * n_vec_per_row + vector_idx];
      }
      asm("# loop ends here");

      // sum up
      result[row_2 + row_1 * ny] = static_cast<float>(sum4_t(corr));
    }
  }
}