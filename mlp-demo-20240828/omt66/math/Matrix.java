package omt66.math;

import java.util.Random;
import java.util.function.Function;

/**
 * Matrix class that supports basic matrix operations.
 */
public class Matrix {
    private double[][] data;
    int rows;
    int cols;

    public Matrix(int rows, int cols) {
        data = new double[rows][cols];
        this.rows = rows;
        this.cols = cols;
    }

    public Matrix(double[][] data) {
        rows = data.length;
        cols = data[0].length;
        this.data = data;
    }

    /**
     * Matrix addition
     * 
     * @param other
     * @return
     */
    public Matrix add(Matrix other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must be the same during addition");
        }

        Matrix res = new Matrix(rows, cols);

        // C = A + B
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                // cij = aij + bij
                res.data[i][j] = data[i][j] + other.data[i][j];
            }
        }

        return res;
    }

    /**
     * Matrix subtraction
     * 
     * @param other
     * @return
     */
    public Matrix subtract(Matrix other) {
        if (rows != other.rows || cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions must be the same during subtraction");
        }

        Matrix res = new Matrix(rows, cols);

        // C = A - B
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] - other.data[i][j];
            }
        }

        return res;
    }

    /**
     * Matrix multiplication
     * 
     * @param other
     * @return
     */
    public Matrix multiply(Matrix other) {
        if (cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions must match for multiplication");
        }

        Matrix res = new Matrix(rows, other.cols);

        // C = A * B
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < other.cols; j++) {
                double sum = 0;
                for (int k = 0; k < cols; k++) {
                    sum += data[i][k] * other.data[k][j];
                }
                res.data[i][j] = sum;
            }
        }

        return res;
    }

    /**
     * Scalar multiplication
     * 
     * @param scalar
     * @return
     */
    public Matrix multiply(double scalar) {
        Matrix res = new Matrix(rows, cols);

        // B = A * scalar
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = data[i][j] * scalar;
            }
        }

        return res;
    }

    /**
     * Matrix transpose
     * 
     * @return
     */
    public Matrix transpose() {
        Matrix res = new Matrix(cols, rows);

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[j][i] = data[i][j];
            }
        }

        return res;
    }

    /**
     * Inverse of a matrix using Gauss-Jordan elimination
     * 
     * @return
     */
    public Matrix inverse() {
        if (this.rows != this.cols) {
            throw new IllegalArgumentException("Matrix must be square to have an inverse.");
        }

        int n = this.rows;
        Matrix augmented = new Matrix(n, 2 * n);

        // Create augmented matrix [A|I]
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                augmented.data[i][j] = this.data[i][j];
                augmented.data[i][j + n] = (i == j) ? 1 : 0;
            }
        }

        // Perform Gauss-Jordan elimination
        for (int i = 0; i < n; i++) {
            // Find pivot
            int max = i;
            for (int j = i + 1; j < n; j++) {
                if (Math.abs(augmented.data[j][i]) > Math.abs(augmented.data[max][i])) {
                    max = j;
                }
            }

            // Swap maximum row with current row
            double[] temp = augmented.data[i];
            augmented.data[i] = augmented.data[max];
            augmented.data[max] = temp;

            // Make all rows below this one 0 in current column
            for (int j = i + 1; j < n; j++) {
                double c = -augmented.data[j][i] / augmented.data[i][i];
                for (int k = i; k < 2 * n; k++) {
                    if (i == k) {
                        augmented.data[j][k] = 0;
                    } else {
                        augmented.data[j][k] += c * augmented.data[i][k];
                    }
                }
            }
        }

        // Solve equation Ax=b using back substitution
        for (int i = n - 1; i >= 0; i--) {
            for (int j = i - 1; j >= 0; j--) {
                double c = -augmented.data[j][i] / augmented.data[i][i];
                for (int k = 0; k < 2 * n; k++) {
                    augmented.data[j][k] += c * augmented.data[i][k];
                }
            }
        }

        // Normalize row
        for (int i = 0; i < n; i++) {
            double c = 1.0 / augmented.data[i][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented.data[i][j] *= c;
            }
        }

        // Extract inverse matrix
        Matrix inverse = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                inverse.data[i][j] = augmented.data[i][j + n];
            }
        }

        return inverse;
    }

    /**
     * Matrix hadamard product
     * 
     * @param other
     * @return
     */
    public Matrix hadamard(Matrix other) {
        if (this.rows != other.rows || this.cols != other.cols) {
            throw new IllegalArgumentException("Matrix dimensions don't match for Hadamard product");
        }
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = this.data[i][j] * other.data[i][j];
            }
        }
        return result;
    }

    /**
     * Calculate determinant of a matrix
     * 
     * @return
     */
    public double determinant() {
        if (rows != cols) {
            throw new IllegalArgumentException("Matrix must be square to have a determinant.");
        }

        if (rows == 1) {
            return data[0][0];
        }

        if (rows == 2) {
            return data[0][0] * data[1][1] - data[0][1] * data[1][0];
        }

        double det = 0;
        for (int i = 0; i < cols; i++) {
            det += Math.pow(-1, i) * data[0][i] * subMatrix(0, i).determinant();
        }

        return det;
    }

    /**
     * Get submatrix by removing a row and column
     * 
     * @param row
     * @param col
     * @return
     */
    public Matrix subMatrix(int row, int col) {
        if (row < 0 || row >= rows || col < 0 || col >= cols) {
            throw new IllegalArgumentException("Invalid row or column index");
        }

        Matrix sub = new Matrix(rows - 1, cols - 1);
        int r = 0;
        for (int i = 0; i < rows; i++) {
            if (i == row) {
                continue;
            }
            int c = 0;
            for (int j = 0; j < cols; j++) {
                if (j == col) {
                    continue;
                }
                sub.data[r][c] = data[i][j];
                c++;
            }
            r++;
        }

        return sub;
    }

    /**
     * Randomize matrix elements
     */
    public void randomize() {
        Random random = new Random();
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                this.data[i][j] = random.nextGaussian();
            }
        }
    }

    /**
     * Convert matrix to array
     * 
     * @return
     */
    public double[] toArray() {
        double[] arr = new double[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(data[i], 0, arr, i * cols, cols);
        }
        return arr;
    }

    /**
     * Convert array to matrix
     * 
     * @param arr
     * @return
     */
    public static Matrix fromArray(double[] arr) {
        Matrix result = new Matrix(arr.length, 1);
        for (int i = 0; i < arr.length; i++) {
            result.data[i][0] = arr[i];
        }
        return result;
    }

    @Override
    public String toString() {
        String info = "Matrix mxn: " + rows + " x " + cols + "\n";
        info += "[\n";
        for (int i = 0; i < rows; i++) {
            info += " [";
            for (int j = 0; j < cols; j++) {
                info += data[i][j] + " ";
            }
            info += "]\n";
        }
        info += "]";
        return info;
    }

    /**
     * Apply a function to each element of the matrix
     * @param func
     * @return
     */
    public Matrix apply(Function<Double, Double> func) {
        Matrix result = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                result.data[i][j] = func.apply(this.data[i][j]);
            }
        }
        return result;
    }

    // --- Static Methods ---
    /**
     * Add two matrices
     * 
     * @param A
     * @param B
     * @return
     */
    public static Matrix add(Matrix A, Matrix B) {
        return A.add(B);
    }

    /**
     * Subtract two matrices
     * 
     * @param A
     * @param B
     * @return
     */
    public static Matrix subtract(Matrix A, Matrix B) {
        return A.subtract(B);
    }

    /**
     * Multiply two matrices
     * 
     * @param A
     * @param B
     * @return
     */
    public static Matrix multiply(Matrix A, Matrix B) {
        return A.multiply(B);
    }

    /**
     * Multiply a matrix by a scalar
     * 
     * @param A
     * @param scalar
     * @return
     */
    public static Matrix multiply(Matrix A, double scalar) {
        return A.multiply(scalar);
    }

    /**
     * Inverse of a matrix
     * 
     * @param A
     * @return
     */
    public static Matrix inverse(Matrix A) {
        return A.inverse();
    }

    /**
     * Hadamard product of two matrices
     * 
     * @param A
     * @param B
     * @return
     */
    public static Matrix hadamard(Matrix A, Matrix B) {
        return A.hadamard(B);
    }

    /**
     * Randomize a matrix
     * 
     * @param rows
     * @param cols
     * @return
     */
    public static Matrix random(int rows, int cols) {
        Matrix res = new Matrix(rows, cols);
        res.randomize();
        return res;
    }

    /**
     * Identity matrix
     * 
     * @param n
     * @return
     */
    public static Matrix identity(int n) {
        Matrix res = new Matrix(n, n);
        for (int i = 0; i < n; i++) {
            res.data[i][i] = 1;
        }
        return res;
    }

    /**
     * Initialize a matrix with a given value
     * 
     * @param rows
     * @param cols
     * @param initialValue
     * @return
     */
    public static Matrix init(int rows, int cols, double initialValue) {
        Matrix res = new Matrix(rows, cols);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                res.data[i][j] = initialValue;
            }
        }
        return res;
    }

    /**
     * Create a matrix of zeros
     * 
     * @param rows
     * @param cols
     * @return
     */
    public static Matrix zeros(int rows, int cols) {
        return Matrix.init(rows, cols, 0);
    }

    /**
     * Create a matrix of ones
     * 
     * @param rows
     * @param cols
     * @return
     */
    public static Matrix ones(int rows, int cols) {
        return Matrix.init(rows, cols, 1);
    }

    /**
     * Solve a system of linear equations Ax = b
     * 
     * @param A
     * @param b
     * @return
     */
    public static Matrix solve(Matrix A, Matrix b) {
        return Matrix.multiply(Matrix.inverse(A), b);
    }

}
