#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

float* load_image(const char* filename, int width, int height) {
    size_t num_pixels = (size_t)width * height;

    // Allocate memory for image
    float* image = malloc(num_pixels * sizeof(float));

    if (image == NULL) {
        fprintf(stderr, "Memory allocation failed!\n");
        return NULL;
    }

    // Open binary file
    FILE* file = fopen(filename, "rb");

    if (file == NULL) {
        perror("Cannot open image");
        free(image);
        return NULL;
    }

    // Read float32 pixel values
    size_t read_count = fread(
        image,
        sizeof(float),
        num_pixels,
        file
    );

    fclose(file);

    if (read_count != num_pixels) {
        fprintf(stderr, "Failed to read image data!\n");
        free(image);
        return NULL;
    }

    return image;
}

int main(void) {
    int width = 512;
    int height = 512;

    float* image = load_image(
        "/home/vitquay1708/Study_Space/Parallel_Computing/data/processed/512/image_01.bin",
        width,
        height
    );

    if (image == NULL) {
        return EXIT_FAILURE;
    }

    printf("Image loaded successfully!\n");
    printf("Dimensions: %d x %d\n", width, height);

    // Access pixel at (x=100, y=200)
    int x = 100;
    int y = 200;

    float pixel = image[y * width + x];

    printf("Pixel (%d, %d): %.2f\n", x, y, pixel);

    free(image);

    return EXIT_SUCCESS;
}