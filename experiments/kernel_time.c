/* kernel_time.c: the clock of experiments/kernel_time.py (the PyTorch
 * baseline at 0.9.0, D87).
 *
 * `kernel_time.py instrument model.c` writes a copy of an exported
 * program's model.c in which every launch, a kernel's call or the OpenMP
 * region that divides it, is timed on the thread that makes it:
 * kt_launch(K, seconds) after it, K the kernel's number in kernels.c. The
 * kernels themselves are not touched, so they compile as in the program
 * (-finstrument-functions changed their code: the course GPT's forward pass
 * under its plan ran 18 % slower with empty hooks). This file keeps every
 * launch in order and writes them at exit, one `K seconds` line each, to
 * the file KERNEL_TIME names (standard error without it); the program's
 * bits do not change.
 */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

double kt_now(void);
void kt_launch(int kernel, double seconds);

struct launch {
    int kernel;
    double seconds;
};

static struct launch *launches;
static size_t count, room;

double kt_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

void kt_launch(int kernel, double seconds)
{
    if (count == room) {
        room = room ? 2 * room : 65536;
        launches = realloc(launches, room * sizeof *launches);
        if (!launches) {
            fprintf(stderr, "kernel_time: out of memory\n");
            exit(1);
        }
    }
    launches[count].kernel = kernel;
    launches[count].seconds = seconds;
    count++;
}

__attribute__((destructor)) static void kt_report(void)
{
    const char *path = getenv("KERNEL_TIME");
    FILE *f = path ? fopen(path, "w") : stderr;
    if (!f)
        return;
    for (size_t i = 0; i < count; i++)
        fprintf(f, "%d %.9f\n", launches[i].kernel, launches[i].seconds);
    if (path)
        fclose(f);
}
