
#include <cmath>
#include <vector>
#include <numeric>
#include <algorithm>

void correlate(int ny, int nx, const float* data, float* result) {
    std::vector<double> matrix(nx * ny);
    
    for (int row = 0; row < ny ; row ++)
    {
        double sum = 0.0;
        
        for (int col = 0; col < nx; col ++)
        {
            sum += data[col + row * nx];
        }

        double mean = sum / nx;

        double square_sum = 0.0;
        
        for (int col = 0; col < nx; col ++)
        {
            double x = data[col + row * nx] - mean;
            matrix[col + row * nx] = x;
            square_sum += x * x;
        }
        
        square_sum = std::sqrt(square_sum);
        
        for(int col = 0; col < nx; col++) {
            matrix[col + row * nx] /= square_sum;
        }
    }
    
    for (int row_1 = 0; row_1 < ny ; row_1++)
    {
        for (int row_2 = row_1; row_2 < ny; row_2++) 
        {
            double corr = 0.0;
            for (int col = 0; col < nx; col++)
            {
                corr += matrix[col + row_1 * nx] * matrix[col + row_2 * nx];
            }
            result[row_2 + row_1 * ny] = static_cast<float>(corr);
        }
    }
}