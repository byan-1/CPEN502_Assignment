package Assignment_1;

import org.junit.Assert;
import org.junit.Test;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

public class NeuralNetTest {
    NeuralNet test = new NeuralNet(
            2,
            4,
            0,
            0.0,
            -1,
            1,
            false
    );

    @Test
    public void testSigmoid() {
        double expectedOutput = 0.622459;
        double actualOutput = test.sigmoid(0.5);
        Assert.assertEquals(expectedOutput, actualOutput, 0.001);
        double expectedCustomOutput = 0.2448197;
        double actualCustomOutput = test.customSigmoid(0.5);
    }

    @Test
    public void testLoadandSave() throws IOException {
        String fileName = "TestWeights1";
        File weights1 = new File(fileName);
        weights1.createNewFile();
        NeuralNet test1 = new NeuralNet(
                2,
                4,
                0,
                0.0,
                -1,
                1,
                false
        );
        NeuralNet test2 = new NeuralNet(
                2,
                4,
                0,
                0.0,
                -1,
                1,
                false
        );
        test1.initializeWeights();
        test2.initializeWeights();
        test1.save(weights1);
        test2.load(fileName);
        double[] hiddentoOut1 = test1.getHiddenToOutWeights();
        double[] hiddentoOut2 = test2.getHiddenToOutWeights();
        double[][] intoHidden1 = test1.getInputToHiddenWeights();
        double[][] intoHidden2 = test2.getInputToHiddenWeights();
        for (int i = 0; i < hiddentoOut1.length; i++) {
            Assert.assertEquals(hiddentoOut1[i], hiddentoOut2[i], 0.000001);
        }
        for (int i = 0; i < intoHidden1.length; i++) {
            for (int j = 0; j < intoHidden1[i].length; j++) {
                Assert.assertEquals(intoHidden1[i][j], intoHidden2[i][j], 0.000001);
            }
        }
    }

    @Test(expected = IOException.class)
    public void testLoadException() throws IOException {
        String fileName = "TestException1";
        File weights1 = new File(fileName);
        weights1.createNewFile();
        NeuralNet test1 = new NeuralNet(
                2,
                6,
                0,
                0.0,
                -1,
                1,
                false
        );
        NeuralNet test2 = new NeuralNet(
                2,
                4,
                0,
                0.0,
                -1,
                1,
                false
        );
        test1.initializeWeights();
        test2.initializeWeights();
        test1.save(weights1);
        test2.load(fileName);
    }
}
