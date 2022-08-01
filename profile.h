#ifndef SYMBOL_profile_1658767998
#define SYMBOL_profile_1658767998

#include <stdio.h>
#include <omp.h>

typedef void(*profile_end)();

void profile(void (*function_to_profile)(profile_end)) {
    double start_time, end_time;

    void finish_profile() {
        end_time = omp_get_wtime();
    }

    printf("\nStarting Profile ...\n");

    start_time = omp_get_wtime();
    function_to_profile(finish_profile);

    printf("Profile Ended in %g\n\n", end_time - start_time);
}

#endif
