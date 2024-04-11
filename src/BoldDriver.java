import java.io.IOException;
import java.util.ArrayList;

public class BoldDriver extends BackPropagation {
    private double[][] input_hiddenWeightsPre;//to store the previous state of the weights on the inputs
    private double[] hiddenLayerBiasesPre;// to store the previous states of the biases on the hidden weights on the inputs
    private double[] hidden_outputWeightsPre;// array to store the previous weights on the output sof the hidden nodes in the MLP
    private double outputBiasPre;// store the previous bias on the output node in the MLP

    public BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);


        System.out.println(learningParameter);
        //System.out.print("MSE: "+String.valueOf(calculateMSE()));
    }

    public BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> dataset1, int nodes, int epochs, Weights_Biases wB) throws IOException {
        super(dataset, dataset1, nodes, epochs, wB);
    }

    @Override
    public void training(int epochs){
        /*This function overrides the Backpropagation model function as it implements annealing features/improvements
         * parameters:
         *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
        double preMSE = calculateMSE(validationDataset);
        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            }
            for (ArrayList<Double> sample : trainingDataset) {
                updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
                //every 1000 epochs update/change the learning parameter of the model.
                if (e % 1000 == 0){
                    if (calculateMSE(trainingDataset) > (preMSE *1.04)){
                        if (learningParameter*0.7 >= 0.01){
                            revertWeights();
                            //decrease learning parameter by 30%.
                            learningParameter = learningParameter*0.7;
                        }
                    } else if (calculateMSE(trainingDataset) < preMSE) {
                        //increase learning parameter by 5% if it does not go past the limit of 0.5
                        if (learningParameter*1.05 <= 0.5){
                            learningParameter *= 1.05;
                        }
                    }
                    storeWeights();
                    preMSE = calculateMSE(validationDataset);
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

}
