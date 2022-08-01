#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <time.h>
#include <omp.h>
#include "profile.h"

#define ARR_SIZE 200000000

static int arr[ARR_SIZE];

void init_dataset() {
    srand(time(NULL));

#pragma omp parallel
    for (int i = 0; i < ARR_SIZE; ++i)
        arr[i] = rand() % 100;
}

void compute_sequentially(profile_end finish_profile) {
    int max_elm = INT_MIN, min_elm = INT_MAX;

    for (int i = 0; i < ARR_SIZE; ++i) {
        min_elm = min_elm < arr[i] ? min_elm : arr[i];
        max_elm = max_elm > arr[i] ? max_elm : arr[i];
    }

    finish_profile();

    printf("Sequential Computation\n");
    printf("Max Value = %d, Min Value = %d\n", max_elm, min_elm);
}

void compute_with_simd_construct(profile_end finish_profile) {
    int max_elm = INT_MIN, min_elm = INT_MAX;

#pragma omp simd
    for (int i = 0; i < ARR_SIZE; ++i) {
        min_elm = min_elm < arr[i] ? min_elm : arr[i];
        max_elm = max_elm > arr[i] ? max_elm : arr[i];
    }

    finish_profile();

    printf("SIMD Construct Computation\n");
    printf("Max Value = %d, Min Value = %d\n", max_elm, min_elm);
}

void compute_with_parallel_sections(profile_end finish_profile) {
    int max_elm = INT_MIN, min_elm = INT_MAX;

#pragma omp parallel sections
    {
#pragma omp section 
        for (int i = 0; i < ARR_SIZE; ++i)
            min_elm = min_elm < arr[i] ? min_elm : arr[i];

#pragma omp section 
        for (int i = 0; i < ARR_SIZE; ++i)
            max_elm = max_elm > arr[i] ? max_elm : arr[i];
    }

    finish_profile();

    printf("Parallel with Sections Computation\n");
    printf("Max Value = %d, Min Value = %d\n", max_elm, min_elm);
}

void compute_with_parallel_for_reduction(profile_end finish_profile) {
    int max_elm = INT_MIN, min_elm = INT_MAX;

#pragma parallel for reduction(min: min_elm) reduction(max: max_elm)
    for (int i = 0; i < ARR_SIZE; ++i) {
        min_elm = min_elm < arr[i] ? min_elm : arr[i];
        max_elm = max_elm > arr[i] ? max_elm : arr[i];
    }

    finish_profile();

    printf("Parallel with Loop Reduction Computation\n");
    printf("Max Value = %d, Min Value = %d\n", max_elm, min_elm);
}

void manual_compute_with_parallel_sections() {
    int max_elm = INT_MIN, min_elm = INT_MAX;
    double time_start, time_end;

#pragma omp parallel sections private(time_start, time_end)
    {
#pragma omp section 
        {
            time_start = omp_get_wtime();
            for (int i = 0; i < ARR_SIZE; ++i)
                min_elm = min_elm < arr[i] ? min_elm : arr[i];
            time_end = omp_get_wtime();
            printf("Time Taken for Min = %g\n", time_end - time_start);
        }

#pragma omp section 
        {
            time_start = omp_get_wtime();
            for (int i = 0; i < ARR_SIZE; ++i)
                max_elm = max_elm > arr[i] ? max_elm : arr[i];
            time_end = omp_get_wtime();
            printf("Time Taken for Max = %g\n", time_end - time_start);
        }
    }

    printf("Parallel with Sections Computation\n");
    printf("Max Value = %d, Min Value = %d\n", max_elm, min_elm);
}

void main(int argc) {
    printf("19BCE0397\tRitvik Gupta\n");

    init_dataset();

    if (argc <= 1) {
        manual_compute_with_parallel_sections();
    } else {
        profile(compute_sequentially);
        profile(compute_with_simd_construct);
        profile(compute_with_parallel_sections);
        profile(compute_with_parallel_for_reduction);
    }
}

