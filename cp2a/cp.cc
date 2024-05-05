
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

void correlate(int ny, int nx, const float *data, float *result)
{
    std::vector<double> matrix(nx * ny);

    double sum, square_sum, mean, x;

    for (int row = 0; row < ny; row++)
    {
        sum = 0.0;

        for (int col = 0; col < nx; col++)
        {
            sum += data[col + row * nx];
        }

        mean = sum / nx;

        square_sum = 0.0;

        for (int col = 0; col < nx; col++)
        {
            x = data[col + row * nx] - mean;
            matrix[col + row * nx] = x;
            square_sum += x * x;
        }

        square_sum = std::sqrt(square_sum);

        for (int col = 0; col < nx; col++)
        {
            matrix[col + row * nx] /= square_sum;
        }
    }

    // parallel coefficient
    int batch_size = 4;
    // number of batches
    int n_batch = nx / batch_size;
    // intermidiate result container
    std::vector<double> batch;
    batch.reserve(batch_size);
    // number of left-over elements
    int residual_size = nx % batch_size;
    // left-over elements
    std::vector<double> residual;
    residual.reserve(residual_size);

    double corr;

    double a,b,c,d;

    for (int row_1 = 0; row_1 < ny; row_1++)
    {
        for (int row_2 = row_1; row_2 < ny; row_2++)
        {
            corr = 0.0;

            // calcualte each batch
            for (int batch_idx = 0; batch_idx < n_batch; batch_idx++)
            {   
                asm("# loop starts here");
                // for (int idx = 0; idx < batch_size; idx++)
                // {   
                a = matrix[batch_idx * batch_size + 0 + row_1 * nx] * matrix[batch_idx * batch_size + 0 + row_2 * nx];
                b = matrix[batch_idx * batch_size + 1 + row_1 * nx] * matrix[batch_idx * batch_size + 1 + row_2 * nx];
                c = matrix[batch_idx * batch_size + 2 + row_1 * nx] * matrix[batch_idx * batch_size + 2 + row_2 * nx];
                d = matrix[batch_idx * batch_size + 3 + row_1 * nx] * matrix[batch_idx * batch_size + 3 + row_2 * nx];
                // }
                // append results
                // corr += std::reduce(batch.begin(), batch.begin() + batch_size, 0.0);
                corr += a+b+c+d;
                asm("# loop ends here");
            }

            // calculate the left-over elements
            for (int idx = 0; idx < residual_size; idx++)
            {
                residual[idx] = matrix[nx - residual_size + idx + row_1 * nx] * matrix[nx - residual_size + idx + row_2 * nx];
            }
            corr += std::reduce(residual.begin(), residual.begin() + residual_size, 0.0);

            // sum up
            result[row_2 + row_1 * ny] = static_cast<float>(corr);
        }
    }
}