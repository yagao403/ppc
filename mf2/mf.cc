#include <vector>
#include <algorithm>

using namespace std;

void mf(int ny, int nx, int hy, int hx, const float *in, float *out)
{

#pragma omp parallel for schedule(static, 1)
  // sliding window (moving down first then moving towards right)
  for (int y = 0; y < ny; y++)
  {
    for (int x = 0; x < nx; x++)
    {
      // window boundaries
      int xmin = (x - hx) > 0 ? (x - hx) : 0;
      int ymin = (y - hy) > 0 ? (y - hy) : 0;
      int xmax = (x + hx + 1) > nx ? nx : (x + hx + 1);
      int ymax = (y + hy + 1) > ny ? ny : (y + hy + 1);

      // generate new window
      const int window_size = (xmax - xmin) * (ymax - ymin);
      vector<float> window(window_size);

      // fill in window
      int idx = 0;

      for (int j = ymin; j < ymax; ++j)
      {
        for (int i = xmin; i < xmax; ++i)
        {
          window[idx++] = in[i + nx*j];
        }
      }

      // calculate median
      if (window_size % 2 == 0)
      {
        nth_element(window.begin(), window.begin() + window_size / 2, window.end());
        float left = window[window_size / 2];
        nth_element(window.begin(), window.begin() + window_size / 2 - 1, window.end());
        float right = window[window_size / 2 - 1];
        out[x + nx*y] = (left + right) / 2.0;
      }
      else
      {
        nth_element(window.begin(), window.begin() + window_size / 2, window.end());
        out[x + nx*y] = window[window_size / 2];
      }
    }
  }
}