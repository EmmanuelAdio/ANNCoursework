import java.io.IOException;
import java.util.ArrayList;

public class Annealing extends BackPropagation {
    public Annealing(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        super(dataset, valDataset, nodes, epochs);
    }

    public Annealing(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> dataset1, int nodes, int epochs, Weights_Biases wB) throws IOException {
        super(dataset, dataset1, nodes, epochs, wB);
    }

    @Override
    public void model(int epochs){
        MSEexp = "";
        for(int e = 0; e < epochs; e++){
            double end = 0.01;
            double start = 0.1;

            learningParameter = end + (start - end)*(1 - 1/(1 + Math.exp(10 - ((double) (20 * e) /epochs))));


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
            }

        }
    }
}
