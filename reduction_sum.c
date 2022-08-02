#include <stdio.h>
#include <omp.h>
#include "profile.h"

#define ARR_SIZE 200000000

static int arr[ARR_SIZE];

void init_dataset() {
#pragma omp parallel for
    for (int i = 0; i < ARR_SIZE; ++i)
        arr[i] = 1;
}

void compute_sequentially(profile_end finish_profile) {
    int sum = 0;

    for (int i = 0;i < ARR_SIZE;++i) {
        sum += arr[i];
    }

    finish_profile();

    printf("Sequential Computation\n");
    printf("Total Sum = %d\n", sum);
}

void compute_with_parallel_reduction(profile_end finish_profile) {
    int sum = 0;

#pragma omp parallel for reduction(+: sum)
    for (int i = 0;i < ARR_SIZE;++i) {
        sum += arr[i];
    }

    finish_profile();

    printf("Parallel Reduced Computation\n");
    printf("Total Sum = %d\n", sum);
}

void manually_compute_with_parallel_reduction() {
    int sum = 0;
    double time_start, time_end;

    time_start = omp_get_wtime();

#pragma omp parallel for reduction(+: sum)
    for (int i = 0;i < ARR_SIZE;++i) {
        sum += arr[i];
    }

    time_end = omp_get_wtime();

    printf("Time Taken for Sum = %g\n", time_end - time_start);
    printf("Parallel Reduced Computation\n");
    printf("Total Sum = %d\n", sum);
}

void main(int argc) {
    printf("19BCE0397\tRitvik Gupta\n");

    init_dataset();

    if (argc <= 1) {
        manually_compute_with_parallel_reduction();
    } else {
        profile(compute_sequentially);
        profile(compute_with_parallel_reduction);
    }
}
