import java.io.IOException;
import java.util.ArrayList;

public class BoldDriver extends BackPropagation {
    private double preMSE;
    private double[][] input_hiddenWeightsPre;
    private double[] hiddenLayerBiasesPre;
    private double[] hidden_outputWeightsPre;
    private double outputBiasPre;

    public BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);


        System.out.println(learningParameter);
        //System.out.print("MSE: "+String.valueOf(calculateMSE()));
    }

    public BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> dataset1, int nodes, int epochs, Weights_Biases wB) throws IOException {
        super(dataset, dataset1, nodes, epochs, wB);
    }

    @Override
    public void model(int epochs){
        MSEexp = "";
        //System.out.println(learningParameter);
        //System.out.print("MSE: "+String.valueOf(calculateMSE()));
        preMSE = calculateMSE(validationDataset);
        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexp += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
//                if (calculateMSE(validationDataset) > preMSEVal){
//                    System.out.println("Broke"+Integer.toString(e));
//                    break;
//                } else {
//                    preMSEVal = calculateMSE(validationDataset);
//                }
            }
            for (ArrayList<Double> sample : trainingDataset) {
                updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
                if (e % 1000 == 0){
                    if (calculateMSE(trainingDataset) > (preMSE*1.04)){
                        if (learningParameter*0.7 >= 0.01){
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

                            //decrease learning parameter by 5%.
                            learningParameter = learningParameter*0.7;
                        }
                    } else if (calculateMSE(trainingDataset) < preMSE) {
                        if (learningParameter*1.05 <= 0.5){
                            learningParameter *= 1.05;
                        }
                    }
                }
            }
        }
    }

    @Override
    public void updateWeights(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs) {
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
        super.updateWeights(sample, deltas, outputs);
    }
}
