// Matrix multiplication : CPU code 
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

#define N 2000 // Size of the square matrices

// Function to allocate memory for a matrix
int** allocate_matrix(int rows, int cols) {
    int** matrix = (int**)malloc(rows * sizeof(int*));
    for (int i = 0; i < rows; i++) {
        matrix[i] = (int*)malloc(cols * sizeof(int));
    }
    return matrix;
}

// Function to initialize a matrix with random values
void initialize_matrix(int** matrix, int rows, int cols) {
    for (int i = 0; i< rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = rand() % 10; // Random value between 0 and 9
        }
    }
}

// Function to print a matrix
void print_matrix(int** matrix, int rows, int cols) {
    for (int i = 0; i< rows; i++) {
        for (int j = 0; j< cols; j++) {
            printf("%d ", matrix[i][j]);
        }
        printf("\n");
    }
}


// Function to perform matrix multiplication
void matrix_multiply(int** A, int** B, int** C) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = 0;
            for (int k = 0; k< N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}


int main() {
    // Allocate memory for matrices A, B, and C
    int** A = allocate_matrix(N, N);
    int** B = allocate_matrix(N, N);
    int** C = allocate_matrix(N, N);

    // Initialize matrices A and B with random values
    initialize_matrix(A, N, N);
    initialize_matrix(B, N, N);

    auto cpu_start_time = std::chrono::high_resolution_clock::now();
    // Perform matrix multiplication
    matrix_multiply(A, B, C);
    auto cpu_end_time = std::chrono::high_resolution_clock::now();

    auto cpu_duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(cpu_end_time - cpu_start_time).count();

    printf("Time taken by CPU: %f microseconds", (float) cpu_duration_ns / 1000);

    // Print matrices A, B, and C
    /*
    printf("Matrix A:\n");
    print_matrix(A, N, N);
    printf("\nMatrix B:\n");
    print_matrix(B, N, N);
    printf("\nMatrix C (Result of A * B): \n");
    print_matrix(C, N, N);
    */

   // Free allocated memory
   for (int i = 0; i < N; i++ ) {
    free(A[i]);
    free(B[i]);
    free(C[i]);
   }
   free(A);
   free(B);
   free(C);


return 0;
}

// Time taken by CPU: 56632440.000000 microseconds
//g++ -std=c++11 matrix_multiplication_cpu.cpp
