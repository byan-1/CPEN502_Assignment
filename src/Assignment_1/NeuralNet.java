package Assignment_1;

import Sarb.NeuralNetInterface;

import java.io.*;
import java.util.HashMap;
import java.util.Random;

public class NeuralNet implements NeuralNetInterface {
    private int numInputs;
    private int numHidden;
    private double learningRate;
    private double momentumTerm;
    private double sigmoidLowerBound;
    private double sigmoidUpperBound;
    private boolean isBipolar;
    private double[] hiddenToOutWeights;
    private double[] prevHiddenToOutWeights;
    private double[][] inputToHiddenWeights;
    private double[][] prevInputToHiddenWeights;
    private double[] hiddenLayerValues;
    private double[] deltaHidden;
    public double curError;
    private String IToHHashKey = "InputToHidden";
    private String HToOHashKey = "HiddenToOutput";

    public NeuralNet(int argNumInputs, int argNumHidden, double argLearningRate, double argMomentumTerm, double argA, double argB, boolean argIsBipolar) {
        numInputs = argNumInputs;
        numHidden = argNumHidden;
        learningRate = argLearningRate;
        momentumTerm = argMomentumTerm;
        sigmoidLowerBound = argA;
        sigmoidUpperBound = argB;
        isBipolar = argIsBipolar;
        hiddenLayerValues = new double[numHidden + 1];
        hiddenLayerValues[numHidden] = 1;
        deltaHidden = new double[numHidden];
        curError = 0;
    }

    @Override
    public double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }

    @Override
    public double customSigmoid(double x) {
        if (isBipolar)
            return sigmoidLowerBound + (sigmoidUpperBound - sigmoidLowerBound) / (1 + Math.exp(-x)) ;
        else
            return sigmoid(x);
    }

    @Override
    public void initializeWeights() {
        Random rand = new Random();
        double min = -0.5;
        double max = 0.5;
        //initialize hidden to output layer weights
        hiddenToOutWeights = new double[numHidden + 1];
        prevHiddenToOutWeights = new double [numHidden + 1];
        double randomVal;
        for (int i = 0; i < numHidden + 1; i++) {
            randomVal = min + (max - min) * rand.nextDouble();
            hiddenToOutWeights[i] = randomVal;
            prevHiddenToOutWeights[i] = randomVal;
        }

        //initialize input to hidden layer weights
        inputToHiddenWeights = new double[numInputs + 1][numHidden];
        prevInputToHiddenWeights = new double[numInputs + 1][numHidden];
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHidden; j++) {
                randomVal = min + (max - min) * rand.nextDouble();
                inputToHiddenWeights[i][j] = randomVal;
                prevInputToHiddenWeights[i][j] = randomVal;
            }
        }
    }

    @Override
    public void zeroWeights() {
        for (int i = 0; i < numHidden; i++) {
            hiddenToOutWeights[i] = 0;
            prevHiddenToOutWeights[i] = 0;
        }
        for (int i = 0; i < numInputs; i++) {
            for (int j = 0; j < numHidden; j++) {
                inputToHiddenWeights[i][j] = 0;
                prevInputToHiddenWeights[i][j] = 0;
            }
        }
    }

    @Override
    public double forwardPropagate(double[] X) {
        //input to hidden layer
        double sums[] = new double[numHidden];
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHidden; j++) {
                sums[j] += X[i] * inputToHiddenWeights[i][j];
            }
        }
        for (int i = 0; i < numHidden; i++) {
            hiddenLayerValues[i] = customSigmoid(sums[i]);
        }
        //hidden to output layer
        double weightedSum = 0;
        for (int i = 0; i < numHidden + 1; i++)
            weightedSum += hiddenToOutWeights[i] * hiddenLayerValues[i];
        return customSigmoid(weightedSum);
    }

    @Override
    public double train(double[] X, double argValue) {
        double[] biasedInput = new double[numInputs + 1];
        for (int i = 0; i < numInputs; i++)
            biasedInput[i] = X[i];
        biasedInput[numInputs] = 1;
        double predictedOutput = forwardPropagate(biasedInput);
        double derivative = predictedOutput * (1 - predictedOutput);
        //get output layer delta
        double deltaOutput = derivative * (argValue - predictedOutput);
        //update hidden to output layer weights
        double changeInWeight;
        for (int i = 0; i < numHidden + 1; i++) {
            changeInWeight = hiddenToOutWeights[i] - prevHiddenToOutWeights[i];
            prevHiddenToOutWeights[i] = hiddenToOutWeights[i];
            hiddenToOutWeights[i] += learningRate * deltaOutput * hiddenLayerValues[i] + momentumTerm * changeInWeight;
        }
        //update hidden layer delta
        for (int i = 0; i < numHidden; i++) {
            derivative = hiddenLayerValues[i] * (1 - hiddenLayerValues[i]);
            deltaHidden[i] = derivative * hiddenToOutWeights[i] * deltaOutput;
        }

        //update input to hidden layer weights
        for (int i = 0; i < numInputs + 1; i++) {
            for (int j = 0; j < numHidden; j++) {
                changeInWeight = inputToHiddenWeights[i][j] - prevInputToHiddenWeights[i][j];
                prevInputToHiddenWeights[i][j] = inputToHiddenWeights[i][j];
                inputToHiddenWeights[i][j] += learningRate * deltaHidden[j] * biasedInput[i] + momentumTerm * changeInWeight;
            }
        }

        return predictedOutput - argValue;
    }

    public double[][] getInputToHiddenWeights() {
        return inputToHiddenWeights;
    }

    public double[] getHiddenToOutWeights() {
        return hiddenToOutWeights;
    }

    @Override
    public void load(String argFileName) throws IOException {
        FileInputStream f = new FileInputStream(argFileName);
        ObjectInputStream input = new ObjectInputStream(f);
        HashMap<String, double[][]> inputMap = null;
        try {
            inputMap = (HashMap<String, double[][]>) input.readObject();
        } catch (ClassNotFoundException e) {
            e.printStackTrace();
        }
        if (inputMap.get(IToHHashKey).length != inputToHiddenWeights.length
        || inputMap.get(HToOHashKey)[0].length != hiddenToOutWeights.length
        ) {
            throw new IOException("Incompatible weight lengths");
        }
        inputToHiddenWeights = inputMap.get(IToHHashKey);
        hiddenToOutWeights = inputMap.get(HToOHashKey)[0];
    }

    //save weights to file via object serialization
    @Override
    public void save(File argFile) throws IOException {
        ObjectOutputStream output = new ObjectOutputStream(new FileOutputStream(argFile));
        //create hashmap of weights
        HashMap<String, double[][]> weightMap = new HashMap<>();
        weightMap.put(IToHHashKey, inputToHiddenWeights);
        double[][] hiddenToOutSave = new double[][]{hiddenToOutWeights};
        weightMap.put(HToOHashKey, hiddenToOutSave);
        //save the hashmap in file
        output.writeObject(weightMap);
        output.flush();
        output.close();
    }
}
