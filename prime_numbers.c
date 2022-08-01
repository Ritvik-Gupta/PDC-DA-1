#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "profile.h"

typedef unsigned int uint;
uint generation_limit;

bool is_prime(uint num) {
    for (uint i = 2; i * i <= num; ++i) {
        if (num % i == 0)
            return false;
    }
    return true;
}

void compute_sequentially(profile_end finish_profile) {
    uint total_primes = 0;

    for (uint i = 2; i <= generation_limit; ++i) {
        if (is_prime(i))
            ++total_primes;
    }

    finish_profile();

    printf("Sequential Computation\n");
    printf("Total Primes counted = %u\n", total_primes);
}

void compute_with_parallel_reduction(profile_end finish_profile) {
    uint total_primes = 0;

#pragma omp parallel for ordered reduction(+: total_primes)
    for (uint i = 2; i <= generation_limit; ++i) {
        if (is_prime(i))
            total_primes += 1;
    }

    finish_profile();

    printf("Parallel Computation\n");
    printf("Total Primes counted = %u\n", total_primes);
}

void compute_with_sieve_of_eratosthenes(profile_end finish_profile) {
    bool* is_not_prime = (bool*)calloc(generation_limit + 1, sizeof(bool));

    for (int p = 2; p * p <= generation_limit; p++) {
        if (!is_not_prime[p]) {
            for (int i = p * p; i <= generation_limit; i += p)
                is_not_prime[i] = true;
        }
    }

    uint total_primes = 0;
    for (int i = 2; i <= generation_limit; i++) {
        if (!is_not_prime[i])
            ++total_primes;
    }

    finish_profile();

    printf("Sieve of Eratosthenes Computation\n");
    printf("Total Primes counted = %u\n", total_primes);
}

void compute_with_parallel_sieve_of_eratosthenes(profile_end finish_profile) {
    bool* is_not_prime = (bool*)calloc(generation_limit + 1, sizeof(bool));

    for (int p = 2; p * p <= generation_limit; p++) {
        if (!is_not_prime[p]) {
#pragma omp parallel for
            for (int i = p * p; i <= generation_limit; i += p)
                is_not_prime[i] = true;
        }
    }

    uint total_primes = 0;
    for (int i = 2; i <= generation_limit; i++) {
        if (!is_not_prime[i])
            ++total_primes;
    }

    finish_profile();

    printf("Parallel Sieve of Eratosthenes Computation\n");
    printf("Total Primes counted = %u\n", total_primes);
}

void manually_compute_with_parallel_reduction() {
    uint total_primes = 0;
    double time_start, time_end;

#pragma omp parallel for reduction(+: total_primes) private(time_start, time_end)
    for (uint num = 2; num <= generation_limit; ++num) {
        bool is_prime = true;

        time_start = omp_get_wtime();
        for (uint i = 2; i * i <= num; ++i) {
            if (num % i == 0)
                is_prime = false;
        }
        time_end = omp_get_wtime();

        if (is_prime)
            total_primes += 1;

        printf("Time taken for %d computation = %g\n", num, 1000 * (time_end - time_start));
    }

    printf("\nParallel Computation\n");
    printf("Total Primes counted = %u\n", total_primes);
}

void main(int argc, char* argv []) {
    printf("19BCE0397\tRitvik Gupta\n");

    generation_limit = atoi(argv[1]);

    if (argc <= 2) {
        manually_compute_with_parallel_reduction();
    } else {
        profile(compute_sequentially);
        profile(compute_with_parallel_reduction);
        profile(compute_with_sieve_of_eratosthenes);
        profile(compute_with_parallel_sieve_of_eratosthenes);
    }
}
