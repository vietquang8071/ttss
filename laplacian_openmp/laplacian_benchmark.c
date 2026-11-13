/*
 * Laplacian filtering benchmark (C11 + OpenMP).
 * Input: headerless, row-major little-endian float32 grayscale image.
 * Usage: ./laplacian_benchmark INPUT.bin SIZE [--output OUTPUT.bin]
 * Input size: SIZE x SIZE pixels.
 * Kernel: [0 1 0; 1 -4 1; 0 1 0], zero padding.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(sizeof(float) == 4, "This program requires 32-bit float");

#define REPEATS 10

static int load_image(const char *path, float *image, size_t pixels) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "Cannot open '%s': %s\n", path, strerror(errno));
        return 0;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        perror("fseek");
        fclose(fp);
        return 0;
    }
    long actual_size = ftell(fp);
    if (actual_size < 0) {
        perror("ftell");
        fclose(fp);
        return 0;
    }
    const size_t expected_size = pixels * sizeof(float);
    if ((size_t)actual_size != expected_size) {
        fprintf(stderr,
                "File size mismatch for '%s': got %ld bytes, expected %zu bytes.\n"
                "Check SIZE, and confirm this is a preprocessed float32 .bin file.\n",
                path, actual_size, expected_size);
        fclose(fp);
        return 0;
    }
    rewind(fp);
    size_t count = fread(image, sizeof(float), pixels, fp);
    if (count != pixels) {
        fprintf(stderr, "Read error: got %zu of %zu floats\n", count, pixels);
        fclose(fp);
        return 0;
    }
    fclose(fp);
    return 1;
}

static int save_image(const char *path, const float *image, size_t pixels) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "Cannot write '%s': %s\n", path, strerror(errno));
        return 0;
    }
    size_t count = fwrite(image, sizeof(float), pixels, fp);
    int close_status = fclose(fp);
    int ok = (count == pixels && close_status == 0);
    if (!ok) fprintf(stderr, "Failed to save '%s'\n", path);
    return ok;
}

/* Same per-pixel arithmetic for sequential and parallel implementations. */
static inline float laplacian_pixel(const float *src, int n, int x, int y) {
    const size_t i = (size_t)y * (size_t)n + (size_t)x;
    const float center = src[i];
    const float left  = x > 0     ? src[i - 1] : 0.0f;
    const float right = x + 1 < n ? src[i + 1] : 0.0f;
    const float up    = y > 0     ? src[i - (size_t)n] : 0.0f;
    const float down  = y + 1 < n ? src[i + (size_t)n] : 0.0f;
    return left + right + up + down - 4.0f * center;
}

static void laplacian_sequential(const float *src, float *dst, int n) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            dst[(size_t)y * n + x] = laplacian_pixel(src, n, x, y);
        }
    }
}

static void laplacian_openmp(const float *src, float *dst, int n, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            dst[(size_t)y * n + x] = laplacian_pixel(src, n, x, y);
        }
    }
}

static int compare_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static double median(double *values, int n) {
    qsort(values, (size_t)n, sizeof(values[0]), compare_double);
    return (n % 2) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

static double benchmark_seq(const float *input, float *output, int n) {
    double times[REPEATS];
    laplacian_sequential(input, output, n); /* warm-up */
    for (int r = 0; r < REPEATS; ++r) {
        double t0 = omp_get_wtime();
        laplacian_sequential(input, output, n);
        times[r] = (omp_get_wtime() - t0) * 1000.0;
    }
    return median(times, REPEATS);
}

static double benchmark_omp(const float *input, float *output, int n, int threads) {
    double times[REPEATS];
    laplacian_openmp(input, output, n, threads); /* warm-up */
    for (int r = 0; r < REPEATS; ++r) {
        double t0 = omp_get_wtime();
        laplacian_openmp(input, output, n, threads);
        times[r] = (omp_get_wtime() - t0) * 1000.0;
    }
    return median(times, REPEATS);
}

typedef struct {
    double max_error;
    double rmse;
    int correct;
} Correctness;

static Correctness check_correctness(const float *reference, const float *actual,
                                      size_t pixels) {
    Correctness result = {0.0, 0.0, 1};
    double sum_square = 0.0;
    for (size_t i = 0; i < pixels; ++i) {
        double a = (double)reference[i];
        double b = (double)actual[i];
        double diff = fabs(a - b);
        if (!isfinite(a) || !isfinite(b) || !isfinite(diff)) {
            result.correct = 0;
            result.max_error = INFINITY;
            result.rmse = INFINITY;
            return result;
        }
        if (diff > result.max_error) result.max_error = diff;
        sum_square += diff * diff;
        if (diff > (1e-5 + 1e-5 * fabs(a))) result.correct = 0;
    }
    result.rmse = sqrt(sum_square / (double)pixels);
    return result;
}

int main(int argc, char **argv) {
    if (argc != 3 && argc != 5) {
        fprintf(stderr, "Usage: %s INPUT.bin SIZE [--output OUTPUT.bin]\n", argv[0]);
        return EXIT_FAILURE;
    }
    if (argc == 5 && strcmp(argv[3], "--output") != 0) {
        fprintf(stderr, "Expected '--output OUTPUT.bin'\n");
        return EXIT_FAILURE;
    }
    char *end = NULL;
    errno = 0;
    long parsed = strtol(argv[2], &end, 10);
    if (errno != 0 || end == argv[2] || *end != '\0' || parsed < 1 || parsed > 16384) {
        fprintf(stderr, "Invalid SIZE: %s\n", argv[2]);
        return EXIT_FAILURE;
    }
    const int n = (int)parsed;
    const size_t pixels = (size_t)n * n;
    if (pixels > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "Image is too large\n");
        return EXIT_FAILURE;
    }

    const uint16_t endian_probe = 1;
    if (*(const uint8_t *)&endian_probe != 1) {
        fprintf(stderr, "This loader requires a little-endian machine\n");
        return EXIT_FAILURE;
    }

    float *input = (float *)malloc(pixels * sizeof(float));
    float *seq_output = (float *)malloc(pixels * sizeof(float));
    float *omp_output = (float *)malloc(pixels * sizeof(float));
    if (!input || !seq_output || !omp_output) {
        fprintf(stderr, "Could not allocate %zu bytes per array\n", pixels * sizeof(float));
        free(input); free(seq_output); free(omp_output);
        return EXIT_FAILURE;
    }
    if (!load_image(argv[1], input, pixels)) {
        free(input); free(seq_output); free(omp_output);
        return EXIT_FAILURE;
    }

    omp_set_dynamic(0);
    double t_seq = benchmark_seq(input, seq_output, n);

    /* All data rows include original filename for straightforward CSV aggregation. */
    printf("file,size,version,threads,median_ms,speedup,efficiency_percent,max_abs_error,rmse,correct\n");
    printf("%s,%d,sequential,0,%.9f,1.000000,0.000000,0,0,PASS\n",
           argv[1], n, t_seq);

    const int thread_counts[] = {1, 2, 4, 8, 12, 16, 20};
    int all_correct = 1;
    for (size_t k = 0; k < sizeof(thread_counts) / sizeof(thread_counts[0]); ++k) {
        const int threads = thread_counts[k];
        double t_omp = benchmark_omp(input, omp_output, n, threads);
        Correctness c = check_correctness(seq_output, omp_output, pixels);
        double speedup = t_seq / t_omp;
        double efficiency = 100.0 * speedup / threads;
        printf("%s,%d,openmp,%d,%.9f,%.6f,%.6f,%.9g,%.9g,%s\n",
               argv[1], n, threads, t_omp, speedup, efficiency,
               c.max_error, c.rmse, c.correct ? "PASS" : "FAIL");
        if (!c.correct) all_correct = 0;
    }

    /* Write outside the timed region; output stays signed float32. */
    int saved = 1;
    if (argc == 5) saved = save_image(argv[4], omp_output, pixels);
    free(input);
    free(seq_output);
    free(omp_output);
    return (all_correct && saved) ? EXIT_SUCCESS : EXIT_FAILURE;
}
