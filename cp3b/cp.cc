/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <vector>
#include <math.h>
#include <x86intrin.h>
typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

constexpr float8_t f8zero {
    0, 0, 0, 0, 0, 0, 0, 0
};

static inline float hsum8(float8_t vv) {
    float v = 0;
    for (int i = 0; i < 8; ++i) {
        v += vv[i];
    }
    return v;
}

static inline float8_t sum8(float8_t a, float8_t b) {
    return a + b;
}

void correlate(int ny, int nx, const float *data, float *result) {

    // vectors per input column
    int na = (ny + 8 - 1) / 8;

    // create and initialize input vector
    std::vector<float> normalized(nx * ny);
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = data[i + j * nx];
        }
    }

    // normalize input vector with each row having arithmetic mean of 0
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        float sum_row = 0;
        // calculate the sum of current row
        for (int i = 0; i < nx; i++) {
            sum_row += normalized[i + j * nx];
        }
        // calculate the mean of current row 
        float mean_row = sum_row / nx;
        // substract the mean of row from row columns
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] -= mean_row;
        }
    }

    // normalize the input vector with each row the sum of the squares of the elements being 1
    #pragma omp parallel for
    for (int j = 0; j < ny; j++) {
        float sum_square = 0;
        // calculate the sum of squares in the row
        for (int i = 0; i < nx; i++) {
            sum_square += normalized[i + j * nx] * normalized[i + j * nx];
        }
        // calculate the factor that is used for multiplying each element
        float factor = sqrt(1 / sum_square);
        // mulitply each element with the factor
        for (int i = 0; i < nx; i++) {
            normalized[i + j * nx] = normalized[i + j * nx] * factor;
        }
    }

    // create and initialize normalized padded vector
    std::vector<float8_t> padded(na * nx);
    #pragma omp parallel for
    for (int ja = 0; ja < na; ++ja) {
        for (int i = 0; i < nx; ++i) {
            for (int jb = 0; jb < 8; ++jb) {
                int j = ja * 8 + jb;
                padded[nx * ja + i][jb] = j < ny ? normalized[i + j * nx] : 0;
            }
        }
    }

    // calculate the (upper triangle of the) matrix product result = padded * padded^T
    #pragma omp parallel for schedule(static,1)
    for (int ja = 0; ja < na; ++ja) {
        // calculate the dot product between current rows j and i
        for (int ia = ja; ia < na; ++ia) {
            // initialize variable that stores the sum of results below
            float8_t vv000 = f8zero;
            float8_t vv001 = f8zero;
            float8_t vv010 = f8zero;
            float8_t vv011 = f8zero;
            float8_t vv100 = f8zero;
            float8_t vv101 = f8zero;
            float8_t vv110 = f8zero;
            float8_t vv111 = f8zero;
            // calculate the dot product between rows for (nab / nb) * 8 elements
            for (int k = 0; k < nx; ++k) {
                float8_t a000 = padded[nx * ia + k];
                float8_t b000 = padded[nx * ja + k];
                float8_t a100 = swap4(a000);
                float8_t a010 = swap2(a000);
                float8_t a110 = swap2(a100);
                float8_t b001 = swap1(b000);
                vv000 = sum8(vv000, a000 * b000);
                vv001 = sum8(vv001, a000 * b001);
                vv010 = sum8(vv010, a010 * b000);
                vv011 = sum8(vv011, a010 * b001);
                vv100 = sum8(vv100, a100 * b000);
                vv101 = sum8(vv101, a100 * b001);
                vv110 = sum8(vv110, a110 * b000);
                vv111 = sum8(vv111, a110 * b001);
            }
            float8_t vv[8] = { vv000, vv001, vv010, vv011, vv100, vv101, vv110, vv111 };
            for (int kb = 1; kb < 8; kb += 2) {
                vv[kb] = swap1(vv[kb]);
            }
            // insert the final value to the result
            for (int jb = 0; jb < 8; ++jb) {
                for (int ib = 0; ib < 8; ++ib) {
                    int i = ib + ia*8;
                    int j = jb + ja*8;
                    if (i < ny && j < ny) {
                        result[i + j * ny] = vv[ib^jb][jb];
                    }
                }
            }
        }
    }
}