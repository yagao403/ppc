/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

#include <vector>
#include <algorithm>
#include <numeric>

void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    for (auto row = 0; row < ny; row++)
    {
        for (auto col = 0; col < nx; col++)
        {   
            std::vector<double> window;
            window.reserve((2*hx + 1) * (2*hy + 1));

            for (auto window_row = std::max(0, row - hy); window_row < std::min(row + hy + 1, ny); window_row++) {
              for (auto window_col = std::max(0, col - hx); window_col < std::min(col + hx + 1, nx); window_col++) {
                window.push_back(in[window_col + nx * window_row]);
              }
            }

            size_t window_size = window.size();
            size_t middle_n;

            if (window_size % 2 == 1) {
              middle_n = (window_size - 1) / 2;
              std::nth_element(window.begin(), window.begin() + middle_n, window.end());
              out[col + nx * row] = static_cast<float>(window[middle_n]);
            } else {
              middle_n = window_size / 2;
              std::nth_element(window.begin(), window.begin() + middle_n-1, window.end());
              std::nth_element(window.begin(), window.begin() + middle_n, window.end());
              // std::sort(window.begin(), window.end());
              // out[col + nx * row] = static_cast<float>((window[middle_n] + window[middle_n-1]) / 2.0);
            }

            window.clear();
        }
    }
}