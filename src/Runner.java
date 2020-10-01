import Assignment_1.NeuralNet;

import java.util.Scanner;

public class Runner {
    private boolean isBipolar;
    private NeuralNet testNetwork;
    private double[][] XORTrainingSet = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    private double[] XORTargetSet = {0, 1, 1, 0};
    private double[][] BPXORTrainingSet = {{-1, -1}, {-1, 1}, {1, -1}, {1, 1}};
    private double[] BPXORTargetSet = {-1, 1, 1, -1};
    private double[][] TrainingSet;
    private double[] TargetSet;

    public Runner(boolean argIsBipolar) {
        isBipolar = argIsBipolar;
        if(isBipolar) {
            TrainingSet = BPXORTrainingSet;
            TargetSet = BPXORTargetSet;
        } else {
            TrainingSet = XORTrainingSet;
            TargetSet = XORTargetSet;
        }
        testNetwork = new NeuralNet(
                2,
                4,
                0.2,
                0.0,
                -1,
                1,
                isBipolar
        );
    }

    private int trainNetwork() {
        int numEpochs = 10000;
        int epochsToReachTarget = 0;
        double target = 0.05;
        testNetwork.initializeWeights();
        for (int i = 0; i < numEpochs; i++) {
            double totalErr = 0;
            for (int j = 0; j < TargetSet.length; j++) {
                totalErr += 0.5 * Math.pow(testNetwork.train(TrainingSet[j], TargetSet[j]), 2);
            }
            if (totalErr < target) {
                epochsToReachTarget = i;
                System.out.println("Target error of " + target + " after" + epochsToReachTarget + " epochs");
                return epochsToReachTarget;
            }
        }
        System.out.println("Did not converge after " + numEpochs + " epochs");
        return -1;
    }

    //plot graphs of training trials for report
    private void trainAndPlot() {

    }
    
    public static void main(String[] args) {
        //Scanner reader = new Scanner(System.in);
        //System.out.print("Enter the number of trials you want to run: ");
        //int numTrials = reader.nextInt();
        int numConverged = 0;
        int sum = 0;
        int epochs = 0;
        for (int i = 0; i < 100; i++) {
            Runner test = new Runner(false);
            epochs = test.trainNetwork();
            if (epochs != -1) {
                numConverged++;
                sum += epochs;
            }
        }
        if (numConverged > 0)
            System.out.println("Average convergence rate: " + (int) sum / numConverged);
    }
}
