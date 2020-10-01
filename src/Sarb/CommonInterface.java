package Sarb;

import java.io.File;
import java.io.IOException;

public interface CommonInterface {
    public double forwardPropagate(double[] X);

    public double train(double[] X, double argValue);

    public void save(File argFile) throws IOException;

    public void load(String argFileName) throws IOException;
}
