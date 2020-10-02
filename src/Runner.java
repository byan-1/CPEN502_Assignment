import Assignment_1.NeuralNet;

import java.util.ArrayList;
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
                0.9,
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
                System.out.println("Target error of " + target + " after " + epochsToReachTarget + " epochs");
                return epochsToReachTarget;
            }
        }
        System.out.println("Did not converge after " + numEpochs + " epochs");
        return -1;
    }

    //train network and get 3 plots, 1 in the lower percentile, 1 around the average, and 1 in the upper percentile (based on previous training done)
    private void trainAndPlot(int minEpochs, int maxEpochs, int avgEpochs) {
        int numEpochs = 10000;
        int epochsToReachTarget = 0;
        double target = 0.05;
        testNetwork.initializeWeights();
        ArrayList<Double> errorRates = new ArrayList<Double>();
        for (int i = 0; i < numEpochs; i++) {
            double totalErr = 0;
            for (int j = 0; j < TargetSet.length; j++) {
                totalErr += 0.5 * Math.pow(testNetwork.train(TrainingSet[j], TargetSet[j]), 2);
            }
            errorRates.add(totalErr);
            if (totalErr < target) {
                epochsToReachTarget = i;
                //find a similar training sample near the min/avg/max epochs from the trials and plot it
                if (epochsToReachTarget < minEpochs + 100)
                    LineChart.displayChart("Error Rate vs Epochs (Lower Percentile Sample)", errorRates);
                if (epochsToReachTarget < avgEpochs + 10 && epochsToReachTarget > avgEpochs - 10)
                    LineChart.displayChart("Error Rate vs Epochs (Average Percentile Sample)", errorRates);
                if (epochsToReachTarget > maxEpochs - 100)
                    LineChart.displayChart("Error Rate vs Epochs (Upper Percentile Sample)", errorRates);
                return;
            }
        }
    }
    
    public static void main(String[] args) {
        //Scanner reader = new Scanner(System.in);
        //System.out.print("Enter the number of trials you want to run: ");
        //int numTrials = reader.nextInt();
        int numConverged = 0;
        int sum = 0;
        int epochs = 0;
        int minEpochs = 0;
        int maxEpochs = 0;
        int avgEpochs = 0;
        boolean isBipolar = false;

        for (int i = 0; i < 100; i++) {
            Runner test = new Runner(isBipolar);
            epochs = test.trainNetwork();
            if (epochs != -1) {
                numConverged++;
                sum += epochs;
                if (epochs > maxEpochs || maxEpochs == 0)
                    maxEpochs = epochs;
                if (epochs < minEpochs)
                    minEpochs = epochs;
            }
        }
        if (numConverged > 0) {
            avgEpochs = (int) sum / numConverged;
            System.out.println("Average convergence rate: " + (int) sum / numConverged);

        }

        Runner test = new Runner(isBipolar);
        test.trainAndPlot(minEpochs, maxEpochs, avgEpochs);
    }
}
