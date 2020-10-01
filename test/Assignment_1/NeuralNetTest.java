package Assignment_1;

import org.junit.Assert;
import org.junit.Test;

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
}
