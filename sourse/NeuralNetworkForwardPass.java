// NeuralNetworkForwardPass.java
// Demonstrates a simple forward pass in a neural network
// Course: Artificial Intelligence (QA–4 Assessment)
// Focus: Computation flow only (no training/backpropagation)

public class NeuralNetworkForwardPass {

    // ----- Step 1: Define activation functions -----

    // ReLU activation: returns max(0, x)
    public static double relu(double x) {
        return Math.max(0, x);
    }

    // Sigmoid activation: squashes value into range (0,1)
    public static double sigmoid(double x) {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    // ----- Step 2: Forward Pass -----
    public static double forwardPass(double[] inputs) {
        // Hardcoded weights and biases

        // Input -> Hidden Layer (3 inputs -> 2 hidden neurons)
        double[][] weightsInputHidden = {
            {0.2, 0.4},   // weights from input neuron 1 to hidden neurons
            {0.5, -0.3},  // weights from input neuron 2 to hidden neurons
            {0.1, 0.2}    // weights from input neuron 3 to hidden neurons
        };
        double[] biasHidden = {0.1, -0.2}; // biases for hidden neurons

        // Hidden -> Output Layer (2 hidden neurons -> 1 output neuron)
        double[][] weightsHiddenOutput = {
            {0.3},   // weight from hidden neuron 1 to output
            {0.7}    // weight from hidden neuron 2 to output
        };
        double[] biasOutput = {0.5}; // bias for output neuron

        // ----- Hidden Layer Computation -----
        double[] hiddenInput = new double[2];   // raw values before activation
        double[] hiddenOutput = new double[2];  // values after ReLU activation

        // Calculate weighted sum for each hidden neuron
        for (int j = 0; j < 2; j++) {
            hiddenInput[j] = biasHidden[j]; // start with bias
            for (int i = 0; i < 3; i++) {
                hiddenInput[j] += inputs[i] * weightsInputHidden[i][j];
            }
            hiddenOutput[j] = relu(hiddenInput[j]); // apply ReLU activation
        }

        // ----- Output Layer Computation -----
        double finalInput = biasOutput[0]; // start with output bias
        for (int j = 0; j < 2; j++) {
            finalInput += hiddenOutput[j] * weightsHiddenOutput[j][0];
        }
        double finalOutput = sigmoid(finalInput); // apply Sigmoid activation

        return finalOutput;
    }

    // ----- Step 3: Test the network -----
    public static void main(String[] args) {
        // Example input vector (3 features)
        double[] inputs = {1.0, 0.5, -1.5};

        // Perform forward pass
        double output = forwardPass(inputs);

        // Print final output
        System.out.println("Final Output: " + output);
    }
}
