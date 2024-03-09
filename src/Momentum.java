import java.io.IOException;
import java.util.ArrayList;

public class Momentum extends BackPropagation {
    private double[][] input_hiddenWeightsChange;
    private double[] hiddenLayerBiasesChange;
    private double[] hidden_outputWeightsChange;
    private double outputBiasChange;

    public Momentum(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    public Momentum(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> dataset1, int nodes, int epochs, Weights_Biases wB) throws IOException {
        super(dataset, dataset1, nodes, epochs, wB);
    }

    @Override
    public void updateWeights(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs) {
        /*This function overrides update weights function for the momentum algorithm, it simply stores the previous weights and adds the
        change in weights to the new updated weights
        Parameter:
            - sample(ArrayList<Double>) = the sample from the data set that we are currently cycling through
            - deltas(ArrayList<Double>) = list of delta values calculated from the backwardsPass function
            - outputs(ArrayList<Double>) = list of the activated outputs of the neural network.*/
        double alpha = 0.9;

        //store the previous weights and biases
        double[][] input_hiddenWeightsPre = new double[nodes][inputs];
        double[] hiddenLayerBiasesPre = new double[nodes];
        double[] hidden_outputWeightsPre = new double[nodes];
        
        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeightsPre[i][j] = input_hiddenWeights[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasesPre[j] = hiddenLayerBiases[j];
            hidden_outputWeightsPre[j] = hidden_outputWeights[j];
        }

        double outputBiasPre = outputBias;

        //update the weights and biases
        for (int i = 0; i < input_hiddenWeights.length; i++){
            double[] weights = input_hiddenWeights[i];
            for(int j = 0; j < weights.length; j++){
                input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + learningParameter*deltas.get(i)*sample.get(j)
                        + alpha*input_hiddenWeightsChange[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiases[j] + learningParameter*deltas.get(j)
                    + alpha*hiddenLayerBiasesChange[j];
            hidden_outputWeights[j] = hidden_outputWeights[j] + learningParameter*deltas.get(deltas.size()-1)*outputs.get(j)
                    + alpha*hidden_outputWeightsChange[j];

        }

        outputBias = outputBias + learningParameter*deltas.get(deltas.size()-1)
                + alpha*outputBiasChange;

        //save teh change in the weights and biases for the next update.
        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeightsChange[i][j] = input_hiddenWeights[i][j] - input_hiddenWeightsPre[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasesChange[j] = hiddenLayerBiases[j] - hiddenLayerBiasesPre[j];

            hidden_outputWeightsChange[j] = hidden_outputWeights[j] - hidden_outputWeightsPre[j];
        }

        outputBiasChange = outputBias - outputBiasPre;
    }

    @Override
    public void initialise(Integer inputs, Integer nodes) {
        /*This function overrides the original initialise function to allow the weight changes to also be initialised.
        * parameters:
       *   - inputs(integer) = the number of predictors in the dataset
       *   - nodes(integer) = the number of hidden nodes in the neural network.*/
        super.initialise(inputs, nodes);

        input_hiddenWeightsChange = new double[nodes][inputs];
        hiddenLayerBiasesChange = new double[nodes];
        hidden_outputWeightsChange = new double[nodes];

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeightsChange[i][j] = 0;
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasesChange[j] = 0.0;

            hidden_outputWeightsChange[j] = 0.0;
        }

        outputBiasChange = 0;
    }

    @Override
    public void initialise(Weights_Biases W_B) {
        /*This function overrides the original initialise function to allow the weight changes to also be initialised.
         * parameters:
         *   - inputs(integer) = the number of predictors in the dataset
         *   - nodes(integer) = the number of hidden nodes in the neural network.*/
        super.initialise(W_B);

        input_hiddenWeightsChange = new double[nodes][inputs];
        hiddenLayerBiasesChange = new double[nodes];
        hidden_outputWeightsChange = new double[nodes];

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeightsChange[i][j] = 0;
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasesChange[j] = 0.0;

            hidden_outputWeightsChange[j] = 0.0;
        }

        outputBiasChange = 0;
    }
}
