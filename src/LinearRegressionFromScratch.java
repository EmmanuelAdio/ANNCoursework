import java.io.FileNotFoundException;
import java.util.ArrayList;

public class LinearRegressionFromScratch {

    public static void main(String[] args) throws FileNotFoundException {
        // Specify the paths to your training and testing dataset files
        PredictorData trainingData_File = new PredictorData("src/Datasets/TrainingDataset.txt");
        PredictorData testingData_File = new PredictorData("src/Datasets/TestingDataset.txt");

        // Load datasets
        ArrayList<ArrayList<Double>> trainingData = trainingData_File.getDataset();
        ArrayList<ArrayList<Double>> testingData = testingData_File.getDataset();

        // Calculate coefficients
        double[] coefficients = calculateCoefficients(trainingData);

        // Use coefficients to predict and calculate MSE
        double mse = calculateMSE(testingData, coefficients[0], coefficients[1]);

        System.out.println("Mean Squared Error (MSE) on Testing Dataset: " + mse);
    }

    // Method to calculate the slope and intercept for the linear regression model
    private static double[] calculateCoefficients(ArrayList<ArrayList<Double>> data) {
        double xSum = 0, ySum = 0, xySum = 0, xxSum = 0;
        int n = data.size();

        // Summation for x, y, xy, and x^2 values
        for (ArrayList<Double> point : data) {
            xSum += point.get(0);
            ySum += point.get(1);
            xySum += point.get(0) * point.get(1);
            xxSum += point.get(0) * point.get(0);
        }

        // Calculate the slope (b1) and intercept (b0) using the least squares method
        double slope = (n * xySum - xSum * ySum) / (n * xxSum - xSum * xSum);
        double intercept = (ySum - slope * xSum) / n;

        return new double[]{slope, intercept};
    }

    private static double calculateMSE(ArrayList<ArrayList<Double>> data, double slope, double intercept) {
        double mse = 0;
        for (ArrayList<Double> point : data) {
            double prediction = slope * point.get(0) + intercept;
            mse += Math.pow(prediction - point.get(1), 2);
        }
        mse /= data.size(); // Average the squared differences
        return mse;
    }
}
