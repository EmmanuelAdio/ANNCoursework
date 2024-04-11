import java.io.IOException;
import java.util.ArrayList;

public class Momentum_Annealing_WeightDecay extends Momentum_WeightDecay{
    private double omega;
    private double upsilon;

    public Momentum_Annealing_WeightDecay(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    @Override
    public void training(int epochs) {
        /*This function overrides the Backpropagation model function as it implements annealing features/improvements
         * parameters:
         *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
        calculateOmega();
        for(int e = 0; e < epochs; e++){
            double end = 0.01;
            double start = 0.1;

            learningParameter = end + (start - end)*(1 - 1/(1 + Math.exp(10 - ((double) (20 * e) /epochs))));


            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            }
            for (ArrayList<Double> sample : trainingDataset) {
                updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
            }

        }
    }
    @Override
    public void calculateOmega(){
        /*This function calculates the value of omega using the information about the model's weights and size. */

        //sum of all weights squared
        double sum = 0.0;
        int n = 0;

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                n++;
                sum += input_hiddenWeights[i][j]*input_hiddenWeights[i][j];
            }
        }

        for (int j = 0; j < nodes; j++){
            sum += hiddenLayerBiases[j]*hiddenLayerBiases[j];
            sum += hidden_outputWeights[j]*hidden_outputWeights[j];
            n++;n++;
        }

        sum += outputBias*outputBias;
        n++;

        omega = ((double) 1 /(2*n))*sum;
    }

    @Override
    public void calculateUpsilon(int e){
        /*This calculates teh value of upsilon depending on the epoch number we are iterating at.
         * parameter:
         *   - e(integer) = the epoch we are on in our training cycle.*/
        upsilon = (1/(learningParameter*e));
    }

    @Override
    public ArrayList<Double> backwardPass(ArrayList<Double> sample, ArrayList<Double> outputs) {
        /*This function overrides the backwardPass from the backpropagation class just add the weight decay features/improvements.
         * parameter:
         *   - sample (ArrayList<Double>) = the data sample we are working on from the dataset
         *   - outputs (ArrayList<Double>) = all the outputs from teh nodes in the neural network.*/

        ArrayList<Double> deltas = new ArrayList<Double>();
        //find the delta function for the final output node formula = (C5-U5)(figOutput5(1-sigOutputs5))
        double finalDelta = ((sample.get(sample.size()-1) - outputs.get(outputs.size()-1)) + upsilon*omega)* derivedActivationFunction(outputs.get(outputs.size()-1));

        //calculate the other deltas
        for(int i = 0; i < hidden_outputWeights.length ; i++){
            double delta = (hidden_outputWeights[i])*finalDelta*(derivedActivationFunction(outputs.get(i)));
            deltas.add(delta);
        }

        deltas.add(finalDelta);

        return deltas;
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
}
