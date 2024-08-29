package demo;

import java.util.Arrays;
import java.util.Scanner;

import omt66.nn.MLP;

public class MLPDemo {
    public static void main(String[] args) {
        MLP nn = new MLP(2, 3, 1);

        // Training data (XOR problem)
        double[][] trainingInputs = { { 0, 0 }, { 0, 1 }, { 1, 0 }, { 1, 1 } };
        double[][] trainingOutputs = { { 0 }, { 1 }, { 1 }, { 0 } };

        // Training loop
        int epochs = 10000;
        System.out.println("Training. Number of epochs: " + epochs);
        for (int i = 0; i < epochs; i++) {
            int index = i % trainingInputs.length;
            nn.train(trainingInputs[index], trainingOutputs[index], 0.1);
        }
        System.out.println("Training completed.");

        // Test the trained network
        System.out.println("Testing...");
        for (double[] input : trainingInputs) {
            double[] output = nn.forward(input);
            System.out.println(" * Input: " + Arrays.toString(input) + " Output: " + Arrays.toString(output));
        }
        System.out.println("Testing completed.");

        boolean done = false;
        Scanner scanner = new Scanner(System.in);
        System.out.println("\nTesting the NN interactively (enter 'q' to quit):");
        while (!done) {
            System.out.println("Enter two numbers (0 or 1) separated by a space: ");
            String answer = scanner.nextLine();
            if (answer.equals("q")) {
                done = true;
                continue;
            }
            double[] input = Arrays.stream(answer.split(" ")).mapToDouble(Double::parseDouble).toArray();
            double[] output = nn.forward(input);
            System.out.println("* Predicted output: " + (output[0] > 0.5 ? 1 : 0));
        }
        scanner.close();
        System.out.println("Bye bye...");
    }
}
