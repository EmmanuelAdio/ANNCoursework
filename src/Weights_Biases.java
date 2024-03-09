import java.util.ArrayList;
import java.util.Random;

public class Weights_Biases {
    /*This function just allows for a weights and biases object to be created and allows for models to be tested with the same initial weights. */
    protected double learningParameter;
    protected double[][] input_hiddenWeights;
    protected double[] hiddenLayerBiases;
    protected double[] hidden_outputWeights;
    protected double outputBias;



    public Weights_Biases(ArrayList<ArrayList<Double>> dataset, int nodes){
        ArrayList<ArrayList<Double>> trainingDataset = new ArrayList<>(dataset);

        int inputs = trainingDataset.get(0).size() - 1;
        learningParameter = 0.1;

        input_hiddenWeights = new double[nodes][inputs];
        hiddenLayerBiases = new double[nodes];
        hidden_outputWeights = new double[nodes];


        double rangeMin = (double) -2 / inputs;
        double rangeMax = (double) 2 / inputs;

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                Random random = new Random();
                double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
                input_hiddenWeights[i][j] = randomWeight;
            }
        }

        for (int j = 0; j < nodes; j++){
            Random random = new Random();
            double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
            hiddenLayerBiases[j] = randomWeight;

            Random random1 = new Random();
            double randomWeight1 = rangeMin + (rangeMax - rangeMin) * random1.nextDouble();
            hidden_outputWeights[j] = randomWeight1;
        }

        Random random2 = new Random();
        outputBias = rangeMin + (rangeMax - rangeMin) * random2.nextDouble();
    }
}
