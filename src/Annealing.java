import java.util.ArrayList;

public class Annealing extends BackPropagation {
    public Annealing(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        super(dataset, valDataset, nodes, epochs);
    }
//it overides an inherited method
    @Override
    public void model(int epochs){
        for(int e = 0; e < epochs; e++){
            double end = 0.01;
            double start = 0.1;

            learningParameter = end + (start - end)*(1 - 1/(1 + Math.exp(10 - ((double) (20 * e) /epochs))));


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
}
