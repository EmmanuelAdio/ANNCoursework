import java.util.ArrayList;

public class WeightDecay extends BackPropagation {
    private double omega;
    private double upsilon;


    public WeightDecay(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        super(dataset, valDataset, nodes, epochs);
        calculateOmega();
    }

    @Override
    public void model(int epochs) {
        for(int e = 0; e < epochs; e++){
            calculateUpsilon(e);
            if ((e % 100) == 0 ){
                for (ArrayList<Double> sample : validationDataset) {
                    forwardPass(sample);
                }
            } else {
                for (ArrayList<Double> sample : trainingDataset) {
                    updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
                }
            }

        }
    }

    public void calculateOmega(){
        //sum of all weights squared
        double sum = 0.0;
        int n = 0;


        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                sum += input_hiddenWeights[i][j]*input_hiddenWeights[i][j];
            }
        }
        n += (nodes*inputs);

        for (int j = 0; j < nodes; j++){
            sum += hiddenLayerBiases[j]*hiddenLayerBiases[j];
            sum += hidden_outputWeights[j]*hidden_outputWeights[j];
        }

        n = n + (2 * hiddenLayerBiases.length) + 1;

        sum += outputBias*outputBias;

        omega = ((double) 1 /(2*n))*sum;
    }

    public void calculateUpsilon(int e){
        upsilon = 1/(learningParameter*e);
    }

    @Override
    public ArrayList<Double> backwardPass(ArrayList<Double> sample, ArrayList<Double> outputs) {
        ArrayList<Double> deltas = new ArrayList<Double>();
        //find the delta function for the final output node formula = (C5-U5)(figOutput5(1-sigOutputs5))
        double finalDelta = (sample.get(sample.size()-1) - outputs.get(outputs.size()-1) + upsilon*omega)*derivedSigFunction(outputs.get(outputs.size()-1));

        for(int i = 0; i < hidden_outputWeights.length ; i++){
            double delta = (hidden_outputWeights[i])*finalDelta*(derivedSigFunction(outputs.get(i)));
            deltas.add(delta);
        }

        deltas.add(finalDelta);

        return deltas;
    }
}
