package omt66.nn;

import java.util.function.Function;

import omt66.math.Matrix;

/**
 * Multi-layer perceptron (MLP) neural network.
 */
public class MLP {
    private final int inputSize;
    private final int hiddenSize;
    private final int outputSize;
    private Matrix weightsInputHidden;
    private Matrix weightsHiddenOutput;
    private Matrix biasHidden;
    private Matrix biasOutput;
    private final Function<Double, Double> activationFunction;
    private final Function<Double, Double> derivativeFunction;

    public MLP(int inputSize, int hiddenSize, int outputSize) {
        this.inputSize = inputSize;
        this.hiddenSize = hiddenSize;
        this.outputSize = outputSize;

        this.weightsInputHidden = new Matrix(this.hiddenSize, this.inputSize);
        this.weightsHiddenOutput = new Matrix(this.outputSize, this.hiddenSize);
        this.biasHidden = new Matrix(this.hiddenSize, 1);
        this.biasOutput = new Matrix(this.outputSize, 1);

        this.weightsInputHidden.randomize();
        this.weightsHiddenOutput.randomize();
        this.biasHidden.randomize();
        this.biasOutput.randomize();

        this.activationFunction = x -> 1 / (1 + Math.exp(-x)); // Sigmoid function
        this.derivativeFunction = x -> x * (1 - x); // Derivative of sigmoid
    }

    public double[] forward(double[] input) {
        if (input.length != inputSize) {
            throw new IllegalArgumentException("Input size must match the network's input size");
        }

        Matrix inputLayer = Matrix.fromArray(input);

        // Hidden layer
        Matrix hiddenLayer = weightsInputHidden.multiply(inputLayer);
        hiddenLayer = hiddenLayer.add(biasHidden);
        hiddenLayer = hiddenLayer.apply(activationFunction);

        // Output layer
        Matrix outputLayer = weightsHiddenOutput.multiply(hiddenLayer);
        outputLayer = outputLayer.add(biasOutput);
        outputLayer = outputLayer.apply(activationFunction);

        return outputLayer.toArray();
    }

    public void train(double[] input, double[] target, double learningRate) {
        // Forward propagation
        Matrix inputLayer = Matrix.fromArray(input);
        Matrix hiddenLayer = weightsInputHidden.multiply(inputLayer);
        hiddenLayer = hiddenLayer.add(biasHidden);
        hiddenLayer = hiddenLayer.apply(activationFunction);

        Matrix outputLayer = weightsHiddenOutput.multiply(hiddenLayer);
        outputLayer = outputLayer.add(biasOutput);
        outputLayer = outputLayer.apply(activationFunction);

        // Calculate the error
        Matrix targets = Matrix.fromArray(target);
        Matrix outputErrors = targets.subtract(outputLayer);

        // Calculate gradient for output layer
        Matrix gradients = outputLayer.apply(derivativeFunction);
        gradients = gradients.hadamard(outputErrors);
        gradients = gradients.multiply(learningRate);

        // Calculate deltas for hidden to output weights
        Matrix hiddenTranspose = hiddenLayer.transpose();
        Matrix weightsHiddenOutputDeltas = gradients.multiply(hiddenTranspose);

        // Adjust hidden to output weights and biases
        weightsHiddenOutput = weightsHiddenOutput.add(weightsHiddenOutputDeltas);
        biasOutput = biasOutput.add(gradients);

        // Calculate hidden layer errors
        Matrix weightsHiddenOutputTranspose = weightsHiddenOutput.transpose();
        Matrix hiddenErrors = weightsHiddenOutputTranspose.multiply(outputErrors);

        // Calculate gradient for hidden layer
        Matrix hiddenGradients = hiddenLayer.apply(derivativeFunction);
        hiddenGradients = hiddenGradients.hadamard(hiddenErrors);
        hiddenGradients = hiddenGradients.multiply(learningRate);

        // Calculate deltas for input to hidden weights
        Matrix inputTranspose = inputLayer.transpose();
        Matrix weightsInputHiddenDeltas = hiddenGradients.multiply(inputTranspose);

        // Adjust input to hidden weights and biases
        weightsInputHidden = weightsInputHidden.add(weightsInputHiddenDeltas);
        biasHidden = biasHidden.add(hiddenGradients);
    }

}
