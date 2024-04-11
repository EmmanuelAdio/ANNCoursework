import java.io.IOException;
import java.util.ArrayList;

public class Momentum_BoldDriver extends BackPropagation{
    private double[][] input_hiddenWeightsChange;//this is the 2d array used to store the changes in the
    private double[] hiddenLayerBiasesChange;//this is the array that stores the biases on the hidden nodes in the MLP
    private double[] hidden_outputWeightsChange;//this is the array of weight on the outputs of the hidden node sin th eMLP
    private double outputBiasChange;// this is the bias on the output node.

    private double[][] input_hiddenWeightsPre;//to store the previous state of the weights on the inputs
    private double[] hiddenLayerBiasesPre;// to store the previous states of the biases on the hidden weights on the inputs
    private double[] hidden_outputWeightsPre;// array to store the previous weights on the output sof the hidden nodes in the MLP
    private double outputBiasPre;// store the previous bias on the output node in the MLP

    private double omega;
    private double upsilon;

    public Momentum_BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    public void training(int epochs) {
        /*This function overrides the Backpropagation model function as it implements weight decay features/improvements
         * parameters:
         *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
        //double preMSE = calculateMSE(validationDataset);
        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            } else {
                for (ArrayList<Double> sample : trainingDataset) {
                    updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
                    //every 1000 epochs update/change the learning parameter of the model.
                }
            }

        }
    }

    public void revertWeights(){
        //revert all the weights
        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeights[i][j] = input_hiddenWeightsPre[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiasesPre[j];
            hidden_outputWeights[j] = hidden_outputWeightsPre[j];
        }

        outputBias = outputBiasPre;
    }

    public void storeWeights(){
        input_hiddenWeightsPre = new double[nodes][inputs];
        hiddenLayerBiasesPre = new double[nodes];
        hidden_outputWeightsPre = new double[nodes];

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                input_hiddenWeightsPre[i][j] = input_hiddenWeights[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasesPre[j] = hiddenLayerBiases[j];
            hidden_outputWeightsPre[j] = hidden_outputWeights[j];
        }

        outputBiasPre = outputBias;
    }

    @Override
    public double activationFunction(double output) {
        /*this function changes the activation function of the previous back propagation algorithm uses tan function instead.
         * parameter:
         *   - output(double) =  the output/value to be activated. */
        output = (Math.exp(output)-Math.exp(-output))/(Math.exp(output)+Math.exp(-output));
        return output;
    }

    @Override
    public double derivedActivationFunction(double output) {
        /*this function derives the activation function of the previous back propagation algorithm uses tan function derivative instead.
         * parameter:
         *   - output(double) =  the output/value to be derived. */
        output = 1-(output*output);
        return output;
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

        //save the change in the weights and biases for the next update.
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
}
