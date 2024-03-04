import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BackPropagation {
    final int nodes;
    final int inputs;
    protected final ArrayList<ArrayList<Double>> trainingDataset;
    protected final ArrayList<ArrayList<Double>> validationDataset;

    protected double learningParameter;
    protected double[][] input_hiddenWeights;
    protected double[] hiddenLayerBiases;
    protected double[] hidden_outputWeights;
    protected double outputBias;


    public BackPropagation(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);

        inputs = trainingDataset.get(0).size() - 1;
        initialise(inputs, nodes);

        model(epochs);
        showResults();
    }

    public void model(int epochs){
        for(int e = 0; e < epochs; e++){
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

    public double calculateMSE(){
        double res = 0.0;
        return res;
    }

    public void showResults(){
        int i = 0;
        for (ArrayList<Double> sample : trainingDataset) {
            i++;
            int s = forwardPass(sample).size();
            System.out.println(sample.get(sample.size()-1).toString()+"|"+forwardPass(sample).get(s-1).toString());

            if (i == 10){
                break;
            }
        }
    }

    public ArrayList<Double> forwardPass(ArrayList<Double> sample){
        ArrayList<Double> outputs = new ArrayList<Double>();

        //do the weighted sum of all the hidden node's inputs
        for (int w = 0; w < input_hiddenWeights.length; w++){
            double[] weights = input_hiddenWeights[w];
            double output = 0.0;
            for (int w2 = 0; w2 < weights.length; w2++){
                output = output + (weights[w2]*sample.get(w2));
            }
            //add the bias
            output = output + hiddenLayerBiases[w];
            outputs.add(sigFunction(output));
        }

        //do the weighted sum of the hidden node's outputs to the output node.
        double output = 0.0;
        for (int h = 0; h < hidden_outputWeights.length; h++){
            output = output + (hidden_outputWeights[h]*outputs.get(h));
        }
        //add the bias
        output = output + outputBias;

        outputs.add(sigFunction(output));
        return outputs;
    }

    public double sigFunction(double output){
        output = 1/(1+Math.exp(-output));
        return output;
    }

    public double derivedSigFunction(double output){
            output = output*(1-output);
            return output;
    }


    public ArrayList<Double> backwardPass(ArrayList<Double> sample,ArrayList<Double> outputs){
        ArrayList<Double> deltas = new ArrayList<Double>();
        //find the delta function for the final output node formula = (C5-U5)(figOutput5(1-sigOutputs5))
        double finalDelta = (sample.get(sample.size()-1) - outputs.get(outputs.size()-1))*derivedSigFunction(outputs.get(outputs.size()-1));

        for(int i = 0; i < hidden_outputWeights.length ; i++){
            double delta = (hidden_outputWeights[i])*finalDelta*(derivedSigFunction(outputs.get(i)));
            deltas.add(delta);
        }

        deltas.add(finalDelta);

        return deltas;
    }

    public void updateWeights(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs){

       for (int i = 0; i < input_hiddenWeights.length; i++){
           double[] weights = input_hiddenWeights[i];
           for(int j = 0; j < weights.length; j++){
               input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + learningParameter*deltas.get(i)*sample.get(j);
           }
       }

        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiases[j] + learningParameter*deltas.get(j);
            hidden_outputWeights[j] = hidden_outputWeights[j] + learningParameter*deltas.get(deltas.size()-1)*outputs.get(j);

        }

       outputBias = outputBias + learningParameter*deltas.get(deltas.size()-1);
    }


    //initialise the weights and biases
    public void initialise(Integer inputs, Integer nodes){
        learningParameter = 0.1;

        input_hiddenWeights = new double[nodes][inputs];
        hiddenLayerBiases = new double[nodes];
        hidden_outputWeights = new double[nodes];


        double rangeMin = (double) -2 /inputs;
        double rangeMax = (double) 2 /inputs;

        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                Random random = new Random();
                double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
                input_hiddenWeights[i][j] = randomWeight;
            }
        }

        for (int j = 0; j < nodes; j++){
            Random random = new Random();
            double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
            hiddenLayerBiases[j] = randomWeight;

            Random random1 = new Random();
            double randomWeight1 = rangeMin + (rangeMax - rangeMin) * random1.nextDouble();
            hidden_outputWeights[j] = randomWeight1;
        }

        Random random2 = new Random();
        outputBias = rangeMin + (rangeMax - rangeMin) * random2.nextDouble();

    }

    public void showWeights(){
        System.out.println("This is the new weights");
        for(double[] arrays: input_hiddenWeights){
            System.out.print(","+ Arrays.toString(arrays));
        }
        System.out.println(Arrays.toString(hidden_outputWeights));
        System.out.println(Arrays.toString(hiddenLayerBiases));
        System.out.println(outputBias);
    }

}
