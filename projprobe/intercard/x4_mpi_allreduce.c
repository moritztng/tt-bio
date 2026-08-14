/* RELION's inter-rank reduction, measured: MPI_Allreduce of MPI_DOUBLE over the
 * packed backprojector volume (real+imag+weight, 24 B/voxel).
 * usage: mpirun -n R ./x4_mpi_allreduce <mb1,mb2,...> [reps]                 */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char *list = argc > 1 ? argv[1] : (char *)"209.29,348.34,741.43";
    int reps = argc > 2 ? atoi(argv[2]) : 6;
    char buf[4096];
    strncpy(buf, list, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;

    if (rank == 0)
        printf("# ranks=%d reps=%d  MPI_DOUBLE MPI_SUM MPI_COMM_WORLD\n", size, reps);

    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        double mb = atof(tok);
        size_t n = (size_t)(mb * 1024.0 * 1024.0 / 8.0);
        double *s = (double *)malloc(n * sizeof(double));
        double *r = (double *)malloc(n * sizeof(double));
        if (!s || !r) { fprintf(stderr, "alloc failed\n"); MPI_Abort(MPI_COMM_WORLD, 1); }
        for (size_t i = 0; i < n; i++) s[i] = (double)(i % 97) + rank;
        /* warm up */
        MPI_Allreduce(s, r, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        double best = 1e30, sum = 0.0;
        for (int k = 0; k < reps; k++) {
            MPI_Barrier(MPI_COMM_WORLD);
            double t0 = MPI_Wtime();
            MPI_Allreduce(s, r, n, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            double dt = MPI_Wtime() - t0, mx;
            MPI_Reduce(&dt, &mx, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
            if (rank == 0) { if (mx < best) best = mx; sum += mx; }
        }
        /* correctness: every rank summed i%97 + rank */
        double want = 0.0;
        for (int p = 0; p < size; p++) want += (double)(5 % 97) + p;
        int ok = (r[5] == want);
        if (rank == 0) {
            double bytes = (double)n * 8.0;
            printf("{\"ranks\": %d, \"mb\": %.2f, \"best_ms\": %.3f, \"mean_ms\": %.3f, "
                   "\"alg_GBs\": %.2f, \"ringbus_GBs\": %.2f, \"exact\": %s}\n",
                   size, bytes / 1048576.0, best * 1e3, sum / reps * 1e3,
                   bytes / best / 1e9, 2.0 * (size - 1) / size * bytes / best / 1e9,
                   ok ? "true" : "false");
            fflush(stdout);
        }
        free(s); free(r);
    }
    MPI_Finalize();
    return 0;
}
