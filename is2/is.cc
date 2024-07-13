#include <cmath>
#include <iostream>
#include <numeric>
#include <stdio.h>
#include <vector>

struct Result {
  int y0;
  int x0;
  int y1;
  int x1;
  float outer[3];
  float inner[3];
};

typedef double pixel __attribute__((vector_size(4 * sizeof(double))));

constexpr pixel d40{0.0, 0.0, 0.0, 0.0};

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/

using namespace std;

Result segment(int ny, int nx, const float *data) {
  int image_size = nx * ny;
  vector<pixel> image_padded((nx + 1) * (ny + 1));
  vector<pixel> image(image_size);
  // fill with zero
  fill(image_padded.begin(), image_padded.end(), d40);
  fill(image.begin(), image.end(), d40);

  double optimal = -1;

  int x0_ret = 0, y0_ret = 0, x1_ret = 0, y1_ret = 0;

  // fill image vector
  for (int row = 0; row < ny; row++) {
    for (int col = 0; col < nx; col++) {
      for (int c = 0; c < 3; c++) {
        image[col + nx * row][c] =
            static_cast<double>(data[c + 3 * (col + nx * row)]);
      }
    }
  }

  for (int row = 0; row < ny; row++) {
    for (int col = 0; col < nx; col++) {
      image_padded[(col + 1) + (nx + 1) * (row + 1)] =
          image[col + nx * row] + image_padded[col + (nx + 1) * (row + 1)] +
          image_padded[(col + 1) + (nx + 1) * row] -
          image_padded[col + (nx + 1) * row];
    }
  }

  pixel image_flat = image_padded[nx + (nx + 1) * ny];

// inner
  double inner_optimal = -1;
  int x0_inner = 0, y0_inner = 0, x1_inner = 0, y1_inner = 0;
  for (int height = 1; height <= ny; height++) {
    for (int width = 1; width <= nx; width++) {
      double pixel_x = height * width;
      double pixel_y = image_size - pixel_x;
      pixel_x = 1 / pixel_x;
      pixel_y = 1 / pixel_y;
      for (int y0 = 0; y0 <= ny - height; y0++) {
        for (int x0 = 0; x0 <= nx - width; x0++) {
          int y1 = y0 + height;
          int x1 = x0 + width;
          double best = 0;
          pixel channel_x_sum = image_padded[x1 + (nx + 1) * y1] -
                                  image_padded[x0 + (nx + 1) * y1] -
                                  image_padded[x1 + (nx + 1) * y0] +
                                  image_padded[x0 + (nx + 1) * y0];
          pixel channel_y_sum = image_flat - channel_x_sum;
          pixel best4 = channel_x_sum * channel_x_sum * pixel_x +
                        channel_y_sum * channel_y_sum * pixel_y;
          best = best4[0] + best4[1] + best4[2];
          if (best > inner_optimal) {
            inner_optimal = best;
            y0_inner = y0;
            x0_inner = x0;
            y1_inner = y1;
            x1_inner = x1;
          }
        }
      }
    }
  }
  {
    if (inner_optimal > optimal) {
      optimal = inner_optimal;
      y0_ret = y0_inner;
      x0_ret = x0_inner;
      y1_ret = y1_inner;
      x1_ret = x1_inner;
    }
  }

  double pixel_x = (y1_ret - y0_ret) * (x1_ret - x0_ret);
  double pixel_y = image_size - pixel_x;
  pixel_x = 1 / pixel_x;
  pixel_y = 1 / pixel_y;
  pixel channel_x_sum, channel_y_sum;
  channel_x_sum = image_padded[x1_ret + (nx + 1) * y1_ret] -
                    image_padded[x0_ret + (nx + 1) * y1_ret] -
                    image_padded[x1_ret + (nx + 1) * y0_ret] +
                    image_padded[x0_ret + (nx + 1) * y0_ret];
  channel_y_sum = image_flat - channel_x_sum;
  channel_y_sum *= pixel_y;
  channel_x_sum *= pixel_x;

  Result result{y0_ret,
                x0_ret,
                y1_ret,
                x1_ret,
                {static_cast<float>(channel_y_sum[0]),
                 static_cast<float>(channel_y_sum[1]),
                 static_cast<float>(channel_y_sum[2])},
                {static_cast<float>(channel_x_sum[0]),
                 static_cast<float>(channel_x_sum[1]),
                 static_cast<float>(channel_x_sum[2])}};

  return result;
}