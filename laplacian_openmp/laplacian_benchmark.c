/*
 * Gaussian-Laplacian zero-crossing benchmark (C11 + OpenMP).
 * Input: headerless, row-major little-endian float32 grayscale image.
 * Usage: ./laplacian_benchmark INPUT.bin SIZE [--output SIGNED.bin]
 *        [--edges-output EDGES.bin] [--sigma SIGMA] [--threshold VALUE]
 * Gaussian: separable 5-tap kernel, radius 2, clamped-edge padding.
 * Laplacian: [0 1 0; 1 -4 1; 0 1 0], zero padding.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

_Static_assert(sizeof(float) == 4, "This program requires 32-bit float");

#define REPEATS 10
#define GAUSSIAN_RADIUS 2
#define GAUSSIAN_TAPS (2 * GAUSSIAN_RADIUS + 1)

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

static int parse_nonnegative_finite(const char *option, const char *text,
                                   int strictly_positive, double *value) {
    char *end = NULL;
    errno = 0;
    double parsed = strtod(text, &end);
    if (errno != 0 || end == text || *end != '\0' || !isfinite(parsed)
        || parsed < 0.0 || (strictly_positive && parsed == 0.0)) {
        fprintf(stderr, "Invalid value for %s: %s\n", option, text);
        return 0;
    }
    *value = parsed;
    return 1;
}

static int output_paths_conflict(const char *first, const char *second) {
    if (strcmp(first, second) == 0) return 1;
    struct stat first_stat;
    struct stat second_stat;
    if (stat(first, &first_stat) == 0 && stat(second, &second_stat) == 0
        && first_stat.st_dev == second_stat.st_dev
        && first_stat.st_ino == second_stat.st_ino) {
        return 1;
    }
    return 0;
}

static void make_gaussian_weights(double sigma, float weights[GAUSSIAN_TAPS]) {
    double values[GAUSSIAN_TAPS];
    double sum = 0.0;
    for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
        const int offset = k - GAUSSIAN_RADIUS;
        const double scaled_offset = (double)offset / sigma;
        values[k] = exp(-0.5 * scaled_offset * scaled_offset);
        sum += values[k];
    }
    for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
        weights[k] = (float)(values[k] / sum);
    }
}

static inline int clamp_index(int value, int n) {
    if (value < 0) return 0;
    if (value >= n) return n - 1;
    return value;
}

static void gaussian_horizontal_sequential(const float *src, float *dst, int n,
                                           const float weights[GAUSSIAN_TAPS]) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float value = 0.0f;
            for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
                const int sx = clamp_index(x + k - GAUSSIAN_RADIUS, n);
                value += src[(size_t)y * n + sx] * weights[k];
            }
            dst[(size_t)y * n + x] = value;
        }
    }
}

static void gaussian_horizontal_openmp(const float *src, float *dst, int n,
                                       const float weights[GAUSSIAN_TAPS], int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float value = 0.0f;
            for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
                const int sx = clamp_index(x + k - GAUSSIAN_RADIUS, n);
                value += src[(size_t)y * n + sx] * weights[k];
            }
            dst[(size_t)y * n + x] = value;
        }
    }
}

static void gaussian_vertical_sequential(const float *src, float *dst, int n,
                                         const float weights[GAUSSIAN_TAPS]) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float value = 0.0f;
            for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
                const int sy = clamp_index(y + k - GAUSSIAN_RADIUS, n);
                value += src[(size_t)sy * n + x] * weights[k];
            }
            dst[(size_t)y * n + x] = value;
        }
    }
}

static void gaussian_vertical_openmp(const float *src, float *dst, int n,
                                     const float weights[GAUSSIAN_TAPS], int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            float value = 0.0f;
            for (int k = 0; k < GAUSSIAN_TAPS; ++k) {
                const int sy = clamp_index(y + k - GAUSSIAN_RADIUS, n);
                value += src[(size_t)sy * n + x] * weights[k];
            }
            dst[(size_t)y * n + x] = value;
        }
    }
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

static inline float zero_crossing_pixel(const float *laplacian, int n,
                                        int x, int y, double threshold) {
    if (x <= 0 || y <= 0 || x >= n - 1 || y >= n - 1) return 0.0f;
    const float center = laplacian[(size_t)y * n + x];
    if (center == 0.0f) return 0.0f;
    for (int dy = -1; dy <= 1; ++dy) {
        const int ny = y + dy;
        if (ny <= 0 || ny >= n - 1) continue;
        for (int dx = -1; dx <= 1; ++dx) {
            if (dx == 0 && dy == 0) continue;
            const int nx = x + dx;
            if (nx <= 0 || nx >= n - 1) continue;
            const float neighbor = laplacian[(size_t)ny * n + nx];
            const int opposite_sign = (center > 0.0f && neighbor < 0.0f)
                                   || (center < 0.0f && neighbor > 0.0f);
            if (opposite_sign && fabs((double)center) + fabs((double)neighbor) >= threshold) {
                return 1.0f;
            }
        }
    }
    return 0.0f;
}

static void zero_crossing_sequential(const float *laplacian, float *edges, int n,
                                     double threshold) {
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            edges[(size_t)y * n + x] = zero_crossing_pixel(laplacian, n, x, y, threshold);
        }
    }
}

static void zero_crossing_openmp(const float *laplacian, float *edges, int n,
                                 double threshold, int threads) {
    #pragma omp parallel for schedule(static) num_threads(threads)
    for (int y = 0; y < n; ++y) {
        for (int x = 0; x < n; ++x) {
            edges[(size_t)y * n + x] = zero_crossing_pixel(laplacian, n, x, y, threshold);
        }
    }
}

static void pipeline_sequential(const float *input, float *horizontal, float *smoothed,
                                float *laplacian, float *edges, int n,
                                const float weights[GAUSSIAN_TAPS], double threshold) {
    gaussian_horizontal_sequential(input, horizontal, n, weights);
    gaussian_vertical_sequential(horizontal, smoothed, n, weights);
    laplacian_sequential(smoothed, laplacian, n);
    zero_crossing_sequential(laplacian, edges, n, threshold);
}

static void pipeline_openmp(const float *input, float *horizontal, float *smoothed,
                            float *laplacian, float *edges, int n,
                            const float weights[GAUSSIAN_TAPS], double threshold,
                            int threads) {
    gaussian_horizontal_openmp(input, horizontal, n, weights, threads);
    gaussian_vertical_openmp(horizontal, smoothed, n, weights, threads);
    laplacian_openmp(smoothed, laplacian, n, threads);
    zero_crossing_openmp(laplacian, edges, n, threshold, threads);
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

static double benchmark_seq(const float *input, float *horizontal, float *smoothed,
                            float *laplacian, float *edges, int n,
                            const float weights[GAUSSIAN_TAPS], double threshold) {
    double times[REPEATS];
    pipeline_sequential(input, horizontal, smoothed, laplacian, edges, n, weights, threshold);
    for (int r = 0; r < REPEATS; ++r) {
        double t0 = omp_get_wtime();
        pipeline_sequential(input, horizontal, smoothed, laplacian, edges, n,
                            weights, threshold);
        times[r] = (omp_get_wtime() - t0) * 1000.0;
    }
    return median(times, REPEATS);
}

static double benchmark_omp(const float *input, float *horizontal, float *smoothed,
                            float *laplacian, float *edges, int n,
                            const float weights[GAUSSIAN_TAPS], double threshold,
                            int threads) {
    double times[REPEATS];
    pipeline_openmp(input, horizontal, smoothed, laplacian, edges, n, weights,
                    threshold, threads);
    for (int r = 0; r < REPEATS; ++r) {
        double t0 = omp_get_wtime();
        pipeline_openmp(input, horizontal, smoothed, laplacian, edges, n, weights,
                        threshold, threads);
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

static int edges_match_exactly(const float *reference, const float *actual, size_t pixels) {
    for (size_t i = 0; i < pixels; ++i) {
        if (reference[i] != actual[i]) return 0;
    }
    return 1;
}

static void free_buffers(float *input, float *horizontal, float *smoothed,
                         float *seq_laplacian, float *omp_laplacian,
                         float *seq_edges, float *omp_edges) {
    free(input);
    free(horizontal);
    free(smoothed);
    free(seq_laplacian);
    free(omp_laplacian);
    free(seq_edges);
    free(omp_edges);
}

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr,
                "Usage: %s INPUT.bin SIZE [--output SIGNED.bin] [--edges-output EDGES.bin] "
                "[--sigma SIGMA] [--threshold VALUE]\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *output_path = NULL;
    const char *edges_output_path = NULL;
    double sigma = 1.0;
    double threshold = 10.0;
    for (int i = 3; i < argc; i += 2) {
        if (i + 1 >= argc) {
            fprintf(stderr, "Missing value for option %s\n", argv[i]);
            return EXIT_FAILURE;
        }
        if (strcmp(argv[i], "--output") == 0) {
            output_path = argv[i + 1];
        } else if (strcmp(argv[i], "--edges-output") == 0) {
            edges_output_path = argv[i + 1];
        } else if (strcmp(argv[i], "--sigma") == 0) {
            if (!parse_nonnegative_finite(argv[i], argv[i + 1], 1, &sigma)) {
                return EXIT_FAILURE;
            }
        } else if (strcmp(argv[i], "--threshold") == 0) {
            if (!parse_nonnegative_finite(argv[i], argv[i + 1], 0, &threshold)) {
                return EXIT_FAILURE;
            }
        } else {
            fprintf(stderr, "Unknown option: %s\n", argv[i]);
            return EXIT_FAILURE;
        }
    }

    if (output_path && edges_output_path
        && output_paths_conflict(output_path, edges_output_path)) {
        fprintf(stderr, "--output and --edges-output must refer to different files\n");
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

    float weights[GAUSSIAN_TAPS];
    make_gaussian_weights(sigma, weights); /* one-time setup, outside benchmark timing */

    float *input = (float *)malloc(pixels * sizeof(float));
    float *horizontal = (float *)malloc(pixels * sizeof(float));
    float *smoothed = (float *)malloc(pixels * sizeof(float));
    float *seq_laplacian = (float *)malloc(pixels * sizeof(float));
    float *omp_laplacian = (float *)malloc(pixels * sizeof(float));
    float *seq_edges = (float *)malloc(pixels * sizeof(float));
    float *omp_edges = (float *)malloc(pixels * sizeof(float));
    if (!input || !horizontal || !smoothed || !seq_laplacian || !omp_laplacian
        || !seq_edges || !omp_edges) {
        fprintf(stderr, "Could not allocate %zu bytes per image buffer\n", pixels * sizeof(float));
        free_buffers(input, horizontal, smoothed, seq_laplacian, omp_laplacian,
                     seq_edges, omp_edges);
        return EXIT_FAILURE;
    }
    if (!load_image(argv[1], input, pixels)) {
        free_buffers(input, horizontal, smoothed, seq_laplacian, omp_laplacian,
                     seq_edges, omp_edges);
        return EXIT_FAILURE;
    }

    omp_set_dynamic(0);
    const double t_seq = benchmark_seq(input, horizontal, smoothed, seq_laplacian,
                                       seq_edges, n, weights, threshold);

    /* All data rows include original filename for straightforward CSV aggregation. */
    printf("file,size,version,threads,median_ms,speedup,efficiency_percent,max_abs_error,rmse,correct\n");
    printf("%s,%d,sequential,0,%.9f,1.000000,0.000000,0,0,PASS\n",
           argv[1], n, t_seq);

    const int thread_counts[] = {2, 4, 8, 12, 16};
    int all_correct = 1;
    for (size_t k = 0; k < sizeof(thread_counts) / sizeof(thread_counts[0]); ++k) {
        const int threads = thread_counts[k];
        const double t_omp = benchmark_omp(input, horizontal, smoothed, omp_laplacian,
                                           omp_edges, n, weights, threshold, threads);
        Correctness c = check_correctness(seq_laplacian, omp_laplacian, pixels);
        c.correct = c.correct && edges_match_exactly(seq_edges, omp_edges, pixels);
        const double speedup = t_seq / t_omp;
        const double efficiency = 100.0 * speedup / threads;
        printf("%s,%d,openmp,%d,%.9f,%.6f,%.6f,%.9g,%.9g,%s\n",
               argv[1], n, threads, t_omp, speedup, efficiency,
               c.max_error, c.rmse, c.correct ? "PASS" : "FAIL");
        if (!c.correct) all_correct = 0;
    }

    /* Write outside the timed region; the diagnostic response remains signed. */
    int saved = 1;
    if (output_path) saved = save_image(output_path, omp_laplacian, pixels) && saved;
    if (edges_output_path) saved = save_image(edges_output_path, omp_edges, pixels) && saved;
    free_buffers(input, horizontal, smoothed, seq_laplacian, omp_laplacian,
                 seq_edges, omp_edges);
    return (all_correct && saved) ? EXIT_SUCCESS : EXIT_FAILURE;
}
