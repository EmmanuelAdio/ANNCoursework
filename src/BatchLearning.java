import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

public class BatchLearning extends BackPropagation{
    protected double[][] input_hiddenWeightChangeSum;//this stores teh sum of the changes made in the weights of the inputs to the hidden nodes
    protected double[] hiddenLayerBiasSum;//this stores the sum of the changes made to the hidden layer's biases
    protected double[] hidden_outputWeightChangeSum;//this stores the sum of the weights of the outputs from the hidden nodes in the MLP.
    protected double outputBiasSum;//this stores the sum of the output nodes bias change

    protected int batchSize;// this is the batch size that we will be training our MLP on.
    public BatchLearning(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs, int batchSize) throws IOException {
        resetChangeSum();
        this.epochs = epochs;
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);
        this.batchSize = batchSize;

        //dynamically enter the inputs into the model.
        inputs = trainingDataset.get(0).size() - 1;
        initialise(inputs, nodes);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
        training(epochs);
        exportError();

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
    }

    @Override
    public void training(int epochs){
        /*This is the function that is used to train the MLP.
         * parameter:
         *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
        int batchSizeCount = batchSize;

        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            }

            // Iterate over each sample in the dataset.
            for (ArrayList<Double> sample : trainingDataset) {
                batchSizeCount--;

                // Perform a forward pass to get predictions.
                ArrayList<Double> res = forwardPass(sample);
                // Accumulate weight and bias changes based on the sample.
                sumOfWeightChange(sample,backwardPass(sample,res),res);

                // If a full batch has been processed, or if we're at the end of the dataset, update weights.
                if (batchSizeCount == 0){
                    updateWeights(batchSize);
                    batchSizeCount = batchSize; // Reset the batch counter.
                }
            }
        }
        //update the weights using the leftover values in the dataset
        if (batchSizeCount != batchSize){
            updateWeights(batchSize - batchSizeCount);
        }
    }

    public void sumOfWeightChange(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs){
        /*This is the function that is used to gather the sum of the weight change during the batch processing training.
         * parameter:
         *   - sample(ArrayList<Double>) = the data point we are currently cycling through
         *   - deltas(ArrayList<Double>) = the deltas(gradients) that will be used toc calculate the weight change
         *   - outputs(ArrayList<Double>) =  the activated outputs of the MLP.*/

        // Iterate over input-hidden weights to accumulate changes.
        for (int i = 0; i < input_hiddenWeightChangeSum.length; i++){
            double[] weights = input_hiddenWeightChangeSum[i];
            for(int j = 0; j < weights.length; j++){
                input_hiddenWeightChangeSum[i][j] = input_hiddenWeightChangeSum[i][j] +  deltas.get(i)*sample.get(j);
            }
        }
        // Accumulate changes for hidden layer biases and hidden-output weights
        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasSum[j] = hiddenLayerBiasSum[j] + deltas.get(j);
            hidden_outputWeightChangeSum[j] = hidden_outputWeightChangeSum[j] + deltas.get(deltas.size()-1)*outputs.get(j);

        }

        // Accumulate change for the output bias.
        outputBiasSum = outputBiasSum + deltas.get(deltas.size()-1);
    }

    public void updateWeights(int Size) {
        /*Updates weights and biases with accumulated changes, then resets accumulators.
        parameters:
         - Size(int) = the batch size to be updated.
        */

        // Apply accumulated changes to input-hidden weights.
        for (int i = 0; i < input_hiddenWeights.length; i++){
            double[] weights = input_hiddenWeights[i];
            for(int j = 0; j < weights.length; j++){
                input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + (learningParameter*input_hiddenWeightChangeSum[i][j])/ Size;
            }
        }

        // Apply accumulated changes to hidden layer biases and hidden-output weights.
        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiases[j] + (learningParameter*hiddenLayerBiasSum[j])/ Size;
            hidden_outputWeights[j] = hidden_outputWeights[j] + (learningParameter*hidden_outputWeightChangeSum[j])/ Size;

        }

        // Update the output bias with the accumulated change.
        outputBias = outputBias + (learningParameter*outputBiasSum)/ Size;
        // Reset the accumulators for the next batch.
        resetChangeSum();
    }



    public void resetChangeSum() {
        /* Resets the accumulators for weight and bias changes.*/
        input_hiddenWeightChangeSum = new double[nodes][inputs];
        hiddenLayerBiasSum = new double[nodes];
        hidden_outputWeightChangeSum = new double[nodes];
        outputBiasSum = 0;
    }

    @Override
    public void initialise(Weights_Biases W_B) {
        resetChangeSum(); // Ensure accumulators are reset.
        super.initialise(W_B);
    }

    @Override
    public void initialise(Integer inputs, Integer nodes) {
        resetChangeSum(); // Reset accumulators before initialization.
        super.initialise(inputs, nodes);
    }
    public void exportError() throws IOException {
        /*This function is used to export the models MSE to a text file to be further manipulated in an Excel file.*/
        String filename = "src/Datasets/"+ Integer.toString(inputs) +"_"+ getClass() +"_"+ Integer.toString(nodes) +"_"+ Integer.toString(epochs)+"_batchSize_"+Integer.toString(batchSize)+".txt";
        File export = new File(filename);
        if (export.createNewFile()) {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        } else {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        }
    }

}
