import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BackPropogation {
    private int inputs;
    private int nodes;
    private ArrayList<ArrayList<Double>> trainingDataset = new ArrayList<ArrayList<Double>>();

    private double learningParameter;
    private double[][] input_hiddenWeights;
    private double[] hiddenLayerBiases;
    private double[] hidden_outputWeights;
    private double outputBias;

    private ArrayList<Double> outputs;
    private ArrayList<Double> deltas;
    private ArrayList<Double> errorValues;


    private ArrayList<Double> modelResults = new ArrayList<Double>();;


    public BackPropogation(ArrayList<ArrayList<Double>> dataset, int nodes, int epochs){
        this.nodes = nodes;

        trainingDataset = dataset;
        this.inputs = trainingDataset.get(0).size()-1;
        initialise(inputs, nodes);

        //initial answers! with weights
        System.out.println("This is the original answers with weights");
        model(1);
        System.out.println(modelResults.toString()+"\n\n");

        System.out.println("This is the new weights");
        for(double[] arrays: input_hiddenWeights){
            System.out.print(","+ Arrays.toString(arrays));
        }
        System.out.println(Arrays.toString(hidden_outputWeights));
        System.out.println(Arrays.toString(hiddenLayerBiases));
        System.out.println(outputBias);

        System.out.println("This is the new answers with new weights");
        model(epochs-1);
        System.out.println(modelResults.toString()+"\n\n");
    }

    public void model(int epochs){
        for(int e = 0; e < epochs; e++){
            modelResults.clear();
            for (ArrayList<Double> sample : trainingDataset) {
                forwardPass(sample);
                backwardPass(sample);
                updateWeights(sample);
            }

        }

    }

    public void forwardPass(ArrayList<Double> sample){
        outputs = new ArrayList<Double>();
        ArrayList<Double> sigOutputs = new ArrayList<Double>();

        //do the weighted sum of all the hidden node's inputs
        for (int w = 0; w < input_hiddenWeights.length; w++){
            double[] weights = input_hiddenWeights[w];
            double output = 0.0;
            for (int w2 = 0; w2 < weights.length; w2++){
                output = output + (weights[w2]*sample.get(w2));
            }
            //add the bias
            output = output + hiddenLayerBiases[w];
            outputs.add(output);
        }

        //do the weighted sum of the hidden node's outputs to the output node.
        double output = 0.0;
        for (int h = 0; h < hidden_outputWeights.length; h++){
            output = output + (hidden_outputWeights[h]*outputs.get(h));
        }
        //add the bias
        output = output + outputBias;
        outputs.add(output);
        modelResults.add(output);


        //calculate the sigmoid functions of each of the inputs
        sigFunction();
    }

    public void sigFunction(){
        ArrayList<Double> res = new ArrayList<Double>();
        for(double d : outputs){
           double result = 1/(1+Math.exp(-d));
           res.add(result);
        }
        outputs.clear();
        outputs = (ArrayList<Double>) res.clone();

        sigDerFunction();
    }

    public void sigDerFunction(){
        errorValues = new ArrayList<Double>();
        for(double d : outputs){
            double x = d*(1-d);
            errorValues.add(x);
        }
    }


    public void backwardPass(ArrayList<Double> sample){
        deltas = new ArrayList<Double>();
        //find the delta function for the final output node formula = (C5-U5)(figOutput5(1-sigOutputs5))
        double finalDelta = (sample.get(sample.size()-1) - outputs.get(outputs.size()-1))*(errorValues.get(errorValues.size()-1));

        for(int i = 0; i < hidden_outputWeights.length ; i++){
            double delta = (hidden_outputWeights[i])*finalDelta*(outputs.get(i));
            deltas.add(delta);
        }

        deltas.add(finalDelta);
    }

    public void updateWeights(ArrayList<Double> sample){
       for (int i = 0; i < input_hiddenWeights.length; i++){
           double[] weights = input_hiddenWeights[i];
           for(int j = 0; j < weights.length; j++){
               input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + learningParameter*deltas.get(i)*sample.get(j);
           }

           for (int j = 0; j < nodes; j++){
               hiddenLayerBiases[j] = hiddenLayerBiases[j] + learningParameter*deltas.get(j);
               hidden_outputWeights[j] = hidden_outputWeights[j] + learningParameter*deltas.get(deltas.size()-1)*outputs.get(j);

           }

           outputBias = outputBias + learningParameter*deltas.get(deltas.size()-1);
       }
    }


    //initialise the weights and biases
    public void initialise(Integer inputs, Integer nodes){
        learningParameter = 0.1;

        input_hiddenWeights = new double[nodes][inputs];
        hiddenLayerBiases = new double[nodes];
        hidden_outputWeights = new double[nodes];


        double rangeMin = (double) -2 /inputs;
        double rangeMax = (double) 2 /inputs;

        System.out.println("Initial Weights and biases");
        for (int i = 0; i < nodes; i++ ){
            for (int j = 0; j < inputs; j++){
                Random random = new Random();
                double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
                input_hiddenWeights[i][j] = randomWeight;
            }
        }
        for(double[] arrays: input_hiddenWeights){
            System.out.print(","+ Arrays.toString(arrays));
        }


        for (int j = 0; j < nodes; j++){
            Random random = new Random();
            double randomWeight = rangeMin + (rangeMax - rangeMin) * random.nextDouble();
            hiddenLayerBiases[j] = randomWeight;

            Random random1 = new Random();
            double randomWeight1 = rangeMin + (rangeMax - rangeMin) * random1.nextDouble();
            hidden_outputWeights[j] = randomWeight1;
        }
        System.out.println(Arrays.toString(hidden_outputWeights));
        System.out.println(Arrays.toString(hiddenLayerBiases));

        Random random2 = new Random();
        outputBias = rangeMin + (rangeMax - rangeMin) * random2.nextDouble();
        System.out.println(outputBias);
        System.out.println("End of Initial\n");

    }

}
