import java.io.IOException;
import java.util.ArrayList;

public class BatchLearning extends BackPropagation{
    protected double[][] input_hiddenWeightChangeSum;
    protected double[] hiddenLayerBiasSum;
    protected double[] hidden_outputWeightChangeSum;
    protected double outputBiasSum;

    protected int datasetSize;
    public BatchLearning(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    @Override
    public void model(int epochs){
        MSEexport = "";
        int batchSize = 100;
        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            }
            int size = batchSize;
            for (ArrayList<Double> sample : trainingDataset) {
                size--;

                ArrayList<Double> res = forwardPass(sample);
                sumOfWeightChange(sample,backwardPass(sample,res),res);

                if (size == 0){
                    updateWeights(batchSize);
                    size = batchSize;
                }
            }
            if (size != batchSize){
                updateWeights(batchSize-size);
            }

        }
    }

    public void sumOfWeightChange(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs){
        for (int i = 0; i < input_hiddenWeightChangeSum.length; i++){
            double[] weights = input_hiddenWeightChangeSum[i];
            for(int j = 0; j < weights.length; j++){
                input_hiddenWeightChangeSum[i][j] = input_hiddenWeightChangeSum[i][j] +  deltas.get(i)*sample.get(j);
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiasSum[j] = hiddenLayerBiasSum[j] + deltas.get(j);
            hidden_outputWeightChangeSum[j] = hidden_outputWeightChangeSum[j] + deltas.get(deltas.size()-1)*outputs.get(j);

        }

        outputBiasSum = outputBiasSum + deltas.get(deltas.size()-1);
    }

    public void updateWeights(int size) {
        for (int i = 0; i < input_hiddenWeights.length; i++){
            double[] weights = input_hiddenWeights[i];
            for(int j = 0; j < weights.length; j++){
                input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + (learningParameter*input_hiddenWeightChangeSum[i][j])/size;
            }
        }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiases[j] + (learningParameter*hiddenLayerBiasSum[j])/datasetSize;
            hidden_outputWeights[j] = hidden_outputWeights[j] + (learningParameter*hidden_outputWeightChangeSum[j])/size;

        }

        outputBias = outputBias + (learningParameter*outputBiasSum)/size;
        resetChangeSum();
    }


    public void resetChangeSum(){
        input_hiddenWeightChangeSum = new double[nodes][inputs];
        hiddenLayerBiasSum = new double[nodes];
        hidden_outputWeightChangeSum = new double[nodes];

        outputBiasSum = 0;
    }

    @Override
    public void initialise(Weights_Biases W_B) {
        datasetSize = trainingDataset.size();
        resetChangeSum();
        super.initialise(W_B);
    }

    @Override
    public void initialise(Integer inputs, Integer nodes) {
        datasetSize = trainingDataset.size();
        resetChangeSum();
        super.initialise(inputs, nodes);
    }
}
