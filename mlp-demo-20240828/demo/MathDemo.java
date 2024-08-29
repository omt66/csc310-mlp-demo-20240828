package demo;

import omt66.math.Matrix;

public class MathDemo {

    public static void main(String[] args) {
        double data1[][] = { { 1, 2, 3 }, { 4, 5, 6 } };
        double data2[][] = { { 1, 2, -3 }, { 5, 2, 4 }, { 10, 1, 9 } };
        double data3[][] = { { 1, 3 }, { -4, 2 } };
        double data4[][] = { { 1 }, { 2 } };

        Matrix m1 = new Matrix(data1);
        Matrix m1T = m1.transpose();
        Matrix m2 = m1.multiply(2);

        System.out.println("Matrix = " + m1);
        System.out.println("Transpose matrix = " + m1T);
        System.out.println("Scaled by 2 = " + m2);

        Matrix m3 = new Matrix(data2);
        Matrix m4 = m3.inverse();
        System.out.println("m3 = " + m3);
        System.out.println("Inverse of m3 = " + m4);

        double determinant = m3.determinant();
        System.out.println("Determinant of m3 = " + determinant);

        Matrix m5 = Matrix.random(3, 3);
        System.out.println("Random matrix = " + m5);

        Matrix m6 = Matrix.zeros(3, 3);
        System.out.println("Matrix of zeros = " + m6);

        Matrix m7 = Matrix.ones(3, 3);
        System.out.println("Matrix of ones = " + m7);

        Matrix m8 = Matrix.identity(3);
        System.out.println("Identity matrix = " + m8);

        Matrix A = new Matrix(data3);
        Matrix b = new Matrix(data4);
        Matrix x = Matrix.solve(A, b);
        System.out.println("Solve for x in Ax = b");
        System.out.println("A = " + A);
        System.out.println("b = " + b);
        System.out.println("x=" + x);
    }

}
