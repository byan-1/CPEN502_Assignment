import javax.swing.*;

import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;

import java.util.ArrayList;

public class LineChart extends JFrame {

    private static final long serialVersionUID = 1L;

    public LineChart(String title, ArrayList<Double> data) {
        super(title);
        // Create dataset
        XYSeries dataset = createDataset(data);
        XYSeriesCollection errData = new XYSeriesCollection(dataset);
        // Create chart
        JFreeChart chart = ChartFactory.createXYLineChart(
                title, // Chart title
                "Epochs", // X-Axis Label
                "Total Error", // Y-Axis Label
                errData
        );

        ChartPanel panel = new ChartPanel(chart);
        setContentPane(panel);
    }

    private XYSeries createDataset(ArrayList<Double> data) {

        XYSeries series = new XYSeries("Title");

        for (int i = 0; i < data.size(); i++) {
            series.add(i, (double)data.get(i));
        }
        return series;
    }

    public static void displayChart(String title, ArrayList<Double> data) {
        SwingUtilities.invokeLater(() -> {
            LineChart chart = new LineChart(title, data);
            chart.setAlwaysOnTop(true);
            chart.pack();
            chart.setSize(1280, 960);
            chart.setDefaultCloseOperation(WindowConstants.EXIT_ON_CLOSE);
            chart.setVisible(true);
        });
    }
}  