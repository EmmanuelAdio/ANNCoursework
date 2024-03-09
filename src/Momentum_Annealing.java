import java.io.IOException;
import java.util.ArrayList;

public class Momentum_Annealing extends Momentum{
    /*This class just extends teh momentum class and adds some of the annealing features.*/
    public Momentum_Annealing(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    @Override
    public void model(int epochs){
        /*This function overrides the Backpropagation model function as it implements annealing features/improvements
         * parameters:
         *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
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
}
