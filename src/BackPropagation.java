import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BackPropagation {
    final int nodes;
    final int inputs;
    final int epochs;
    protected final ArrayList<ArrayList<Double>> trainingDataset;
    protected final ArrayList<ArrayList<Double>> validationDataset;

    protected double learningParameter;
    protected double[][] input_hiddenWeights;
    protected double[] hiddenLayerBiases;
    protected double[] hidden_outputWeights;
    protected double outputBias;

    private static final DecimalFormat decFor = new DecimalFormat("0.00000000");

    protected double preMSEVal;

    protected String MSEexp;

    private String results;


    public BackPropagation(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        this.epochs = epochs;
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);

        inputs = trainingDataset.get(0).size() - 1;
        initialise(inputs, nodes);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
        model(epochs);
        showResults(trainingDataset);
        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
    }

    public BackPropagation(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs, Weights_Biases w_B) throws IOException {
        this.epochs = epochs;
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);

        inputs = trainingDataset.get(0).size() - 1;
        initialise(w_B);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
        model(epochs);
        exportError();
        showResults(trainingDataset);
        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
    }

    public void model(int epochs){
        MSEexp = "";
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

            }
        }
    }

    public void exportError() throws IOException {
        String filename = "src/Datasets/"+ getClass() +"_"+ Integer.toString(nodes) +"_"+ Integer.toString(epochs)+".txt";
        File export = new File(filename);
        if (export.createNewFile()) {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexp);
            fWrite.close();
        } else {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexp);
            fWrite.close();
        }
    }

    public double calculateMSE(ArrayList<ArrayList<Double>> testing){

        int n = 0;
        double sum = 0;

        for (ArrayList<Double> sample : testing) {
            n++;
            int s = forwardPass(sample).size();
            double x = sample.get(sample.size()-1) - forwardPass(sample).get(s-1);
            sum += x*x;
        }

        return Double.parseDouble(decFor.format(sum/n));

    }

    public void showResults(ArrayList<ArrayList<Double>> testing){
//        int i = 0;
        results = "";
        for (ArrayList<Double> sample : testing) {
//            i++;
            int s = forwardPass(sample).size();
            results += sample.get(sample.size()-1).toString()+"|"+forwardPass(sample).get(s-1).toString()+"\n";

//            if (i == 10){
//                break;
//            }
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

        preMSEVal = calculateMSE(validationDataset);

    }

    public void initialise(Weights_Biases W_B){
        learningParameter = W_B.learningParameter;

        input_hiddenWeights = new double[nodes][inputs];
        hiddenLayerBiases = new double[nodes];
        hidden_outputWeights = new double[nodes];

        for (int i = 0; i < nodes; i++ ){
            input_hiddenWeights[i] = W_B.input_hiddenWeights[i].clone();
        }

        hiddenLayerBiases = W_B.hiddenLayerBiases.clone();
        hidden_outputWeights = W_B.hidden_outputWeights.clone();

        outputBias = W_B.outputBias;

        preMSEVal = calculateMSE(validationDataset);
        showWeights();
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
