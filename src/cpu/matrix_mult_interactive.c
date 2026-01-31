#include <stdio.h>
#include <stdlib.h>

// Function to read a matrix
void readMatrix(int rows, int cols, int matrix[rows][cols], const char* name) {
    printf("\nEnter elements for %s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("Element [%d][%d]: ", i + 1, j + 1);
            scanf("%d", &matrix[i][j]);
        }
    }
}

// Function to print a matrix
void printMatrix(int rows, int cols, int matrix[rows][cols], const char* name) {
    printf("\n%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d\t", matrix[i][j]);
        }
        printf("\n");
    }
}

// Function to multiply two matrices
void multiplyMatrices(int rows1, int cols1, int matrix1[rows1][cols1],
                      int rows2, int cols2, int matrix2[rows2][cols2],
                      int result[rows1][cols2]) {
    // Initialize result matrix to zero
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            result[i][j] = 0;
        }
    }
    
    // Perform matrix multiplication
    for (int i = 0; i < rows1; i++) {
        for (int j = 0; j < cols2; j++) {
            for (int k = 0; k < cols1; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
}

int main() {
    int rows1, cols1, rows2, cols2;
    
    printf("=== Matrix Multiplication Program ===\n\n");
    
    // Get dimensions of first matrix
    printf("Enter dimensions of first matrix (rows columns): ");
    scanf("%d %d", &rows1, &cols1);
    
    // Get dimensions of second matrix
    printf("Enter dimensions of second matrix (rows columns): ");
    scanf("%d %d", &rows2, &cols2);
    
    // Check if multiplication is possible
    if (cols1 != rows2) {
        printf("\nError: Matrix multiplication not possible!\n");
        printf("Number of columns of first matrix (%d) must equal ", cols1);
        printf("number of rows of second matrix (%d).\n", rows2);
        return 1;
    }
    
    // Declare matrices using variable-length arrays
    int matrix1[rows1][cols1];
    int matrix2[rows2][cols2];
    int result[rows1][cols2];
    
    // Read matrices
    readMatrix(rows1, cols1, matrix1, "Matrix A");
    readMatrix(rows2, cols2, matrix2, "Matrix B");
    
    // Display input matrices
    printMatrix(rows1, cols1, matrix1, "Matrix A");
    printMatrix(rows2, cols2, matrix2, "Matrix B");
    
    // Multiply matrices
    multiplyMatrices(rows1, cols1, matrix1, rows2, cols2, matrix2, result);
    
    // Display result
    printMatrix(rows1, cols2, result, "Result (A × B)");
    
    return 0;
}
