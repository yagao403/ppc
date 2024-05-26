#include <algorithm>
#include <omp.h>
#include <cmath>

typedef unsigned long long data_t;

void quickSort(int thread, int n, data_t *data)
{

    if (n < 2)
        return;
    if (thread <= 0)
    {
        std::sort(data, data + n);
        return;
    }

    // choosing a better pivot
    data_t p1 = data[n / 2];
    data_t p2 = data[0];
    data_t p3 = data[n - 1];
    data_t max = std::max(p2, p3);
    data_t pivot = p1 + p2 + p3 - std::max(p1, max) - std::min(p1, max);

    data_t *left = data;
    data_t *right = data + n - 1;

    while (left <= right)
    {
        while (*left < pivot)
            left++;
        while (*right > pivot)
            right--;
        if (left <= right)
        {
            data_t temp = *left;
            *left = *right;
            *right = temp;
            left++;
            right--;
        }
    }
#pragma omp task shared(data)
    quickSort(thread - 1, right - data + 1, data);
#pragma omp task shared(data)
    quickSort(thread - 1, n - (left - data), left);
}

void psort(int n, data_t *data)
{
    int thread = static_cast<int>((std::log2(omp_get_max_threads())) * 2);
#pragma omp parallel
    {
#pragma omp single
        quickSort(thread, n, data);
#pragma omp taskwait
    }
}