#include <algorithm>
#include <cmath>
#include <omp.h>

typedef unsigned long long data_t;

void mergeSort(int taskCounter, int n, data_t *data) {
    if (taskCounter <= 0) {
        std::sort(data, data + n);
        return;
    }
    
    int pivot = (n + 1) / 2;
    
    #pragma omp task
    mergeSort(taskCounter - 1, pivot, data);
    
    #pragma omp task
    mergeSort(taskCounter - 1, n - pivot, data + pivot);
    
    #pragma omp taskwait
    std::inplace_merge(data, data+pivot, data+n);
}

void psort(int n, data_t *data) {
    int taskCounter = static_cast<int>(std::log2(omp_get_max_threads())) * 2;
    
    #pragma omp parallel
    {
        #pragma omp single
        {
            mergeSort(taskCounter, n, data);
        }
    }
}