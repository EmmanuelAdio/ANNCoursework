import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Random;

public class BackPropagation {
    /*This is the base back propagation algorithm the main backbone of this java project*/
    final int nodes;
    final int inputs;
    final int epochs;
    protected final ArrayList<ArrayList<Double>> trainingDataset;//variable to store the Training dataset
    protected final ArrayList<ArrayList<Double>> validationDataset;//variable to store the validation dataset

    protected double learningParameter;
    protected double[][] input_hiddenWeights;//array to store the weights of the inputs going into the hidden nodes
    protected double[] hiddenLayerBiases;//array to store the biases of the hidden nodes in the neural network.
    protected double[] hidden_outputWeights;// array to store the weights on the outputs from the hidden nodes
    protected double outputBias;// the biases on the output node.

    private static final DecimalFormat decFor = new DecimalFormat("0.00000000");

    protected double preMSEVal;// variable to store the previous MSE on the validation data.

    protected String MSEexport;// store the string that will be exported for the MSE.txt

    private String results;


    public BackPropagation(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        this.epochs = epochs;
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);

        //dynamically enter the inputs into the model.
        inputs = trainingDataset.get(0).size() - 1;
        initialise(inputs, nodes);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
        model(epochs);
        exportError();
        showResults(trainingDataset);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
    }

    public BackPropagation(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs, Weights_Biases w_B) throws IOException {
        this.epochs = epochs;
        this.nodes = nodes;
        trainingDataset = new ArrayList<>(dataset);
        validationDataset = new ArrayList<>(valDataset);

        //dynamically enter the inputs into the model.
        inputs = trainingDataset.get(0).size() - 1;
        initialise(w_B);

        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
        model(epochs);
        exportError();
        showResults(trainingDataset);
        System.out.printf("MSE: "+String.valueOf(calculateMSE(validationDataset))+"\n");
    }

    public void model(int epochs){
        /*This is the function that teh training of the ANN is started.
        * parameter:
        *   - epochs(integer) = the max number of loops through the ANN will perform when training.*/
        MSEexport = "";
        for(int e = 0; e < epochs; e++){
            if ((e % 100) == 0 ){
                MSEexport += Integer.toString(e)+"|"+Double.toString(calculateMSE(validationDataset))+"|"+Double.toString(calculateMSE(trainingDataset))+"\n";
            }
            for (ArrayList<Double> sample : trainingDataset) {
                updateWeights(sample,backwardPass(sample,forwardPass(sample)),forwardPass(sample));
            }
        }
    }

    public void exportError() throws IOException {
        /*This function is used to export the models MSE to a text file to be further manipulated in an Excel file.*/
        String filename = "src/Datasets/"+ Integer.toString(inputs) +"_"+ getClass() +"_"+ Integer.toString(nodes) +"_"+ Integer.toString(epochs)+".txt";
        File export = new File(filename);
        if (export.createNewFile()) {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        } else {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        }
    }

    public void exportResults() throws IOException {
        /*This function is used to export the models results to a text file to be further manipulated in an Excel file.*/
        String filename = "src/Datasets/"+ Integer.toString(inputs) +"_"+ getClass() +"_"+ Integer.toString(nodes) +"_"+ Integer.toString(epochs)+"_results.txt";
        File export = new File(filename);
        if (export.createNewFile()) {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        } else {
            FileWriter fWrite = new FileWriter(filename);
            fWrite.write(MSEexport);
            fWrite.close();
        }
    }

    public double calculateMSE(ArrayList<ArrayList<Double>> testing){
        /*This is the function used to calculate the MSE(mean squared Error) for the current state of the ANN model.
        * parameter:
        *   - testing(ArrayList<ArrayList<Double>>) = this is the data set we will be using to calculate our model MSE.*/
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
        /*This function shows the actual index value with the model's predicted index values
        * parameters:
        *   - testing(ArrayList<ArrayList<Double>>) = this is the data set we will eb collecting actual index flood values from. */
        results = "";
        for (ArrayList<Double> sample : testing) {
            int s = forwardPass(sample).size();
            results += sample.get(sample.size()-1).toString()+"|"+forwardPass(sample).get(s-1).toString()+"\n";
        }
        showResultsConsole(testing);
    }

    public void showResultsConsole(ArrayList<ArrayList<Double>> show){
        /*Unimportant function used to show actual index value with model predicated values
        * parameter:
        *   - show(ArrayList<ArrayList<Double>>) : the dataset that the hopefully trained ANN model will run through and display its predictions compared to the actual values.*/
        int i = 0;
        for (ArrayList<Double> sample : show) {
            i++;
            int s = forwardPass(sample).size();
            System.out.println(sample.get(sample.size()-1).toString()+"         "+forwardPass(sample).get(s-1).toString());
            if (i == 10){
                break;
            }
        }
    }

    public ArrayList<Double> forwardPass(ArrayList<Double> sample){
        /*This is the forwardPass function the second step in the backpropagation algorithm.
        * parameters:
            - sample(ArrayList<Double>) : this is the data sample in the dataset that we are currently on.*/
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
            outputs.add(activationFunction(output));
        }

        //do the weighted sum of the hidden node's outputs to the output node.
        double output = 0.0;
        for (int h = 0; h < hidden_outputWeights.length; h++){
            output = output + (hidden_outputWeights[h]*outputs.get(h));
        }
        //add the bias
        output = output + outputBias;

        outputs.add(activationFunction(output));
        return outputs;
    }

    public double activationFunction(double output){
        /*
        This is the activation function applied on the outputs in the neural network, here we are using the sigmoid function
        parameter:
            - output(double) = is the double value that we are applying the sigmoid function to.
        * */
        output = 1/(1+Math.exp(-output));
        return output;
    }

    public double derivedActivationFunction(double output){
        /*
        This is the derived activation function applied on the outputs in the neural network, here we are deriving the sigmoid function
        parameter:
            - output(double) = is the double value that we are applying the derived sigmoid function to.
        * */
            output = output*(1-output);
            return output;
    }


    public ArrayList<Double> backwardPass(ArrayList<Double> sample,ArrayList<Double> outputs){
        /*
        This is the back pass function the third step in the back propagation algorithm
        parameter:
            - sample(ArrayList<Double>) = this is the data sample in the dataset that we are currently on.
            - outputs(ArrayList<Double>) = is the list of outputs that the neural networks nodes  (hidden and output) produce
        * */
        ArrayList<Double> deltas = new ArrayList<Double>();

        //find the delta function for the final output node formula = (C5-U5)(figOutput5(1-sigOutputs5))
        double finalDelta = (sample.get(sample.size()-1) - outputs.get(outputs.size()-1))* derivedActivationFunction(outputs.get(outputs.size()-1));

        //find the deltas for the hidden nodes in the neural network.
        for(int i = 0; i < hidden_outputWeights.length ; i++){
            double delta = (hidden_outputWeights[i])*finalDelta*(derivedActivationFunction(outputs.get(i)));
            deltas.add(delta);
        }

        deltas.add(finalDelta);

        return deltas;
    }

    public void updateWeights(ArrayList<Double> sample, ArrayList<Double> deltas, ArrayList<Double> outputs){
        /*This is the function that updates all the weights and biases in the neural network
        * parameters:
        *   - sample(ArrayList<Double>) = the sample from the data set that we are currently cycling through
        *   - deltas(ArrayList<Double>) = list of delta values calculated from the backwardsPass function
        *   - outputs(ArrayList<Double>) = list of the activated outputs of the neural network.*/

        //update the weights for the inputs to the hidden nodes
       for (int i = 0; i < input_hiddenWeights.length; i++){
           double[] weights = input_hiddenWeights[i];
           for(int j = 0; j < weights.length; j++){
               input_hiddenWeights[i][j] = input_hiddenWeights[i][j] + learningParameter*deltas.get(i)*sample.get(j);
           }
       }

       //update the weights for the outputs of the hidden node
        //update the biases on the hidden nodes
        for (int j = 0; j < nodes; j++){
            hiddenLayerBiases[j] = hiddenLayerBiases[j] + learningParameter*deltas.get(j);
            hidden_outputWeights[j] = hidden_outputWeights[j] + learningParameter*deltas.get(deltas.size()-1)*outputs.get(j);

        }

        //update the output node's bias.
       outputBias = outputBias + learningParameter*deltas.get(deltas.size()-1);
    }


    //initialise the weights and biases
    public void initialise(Integer inputs, Integer nodes){
        /*This is just the weight initialising function that randomizes the models initial weights
        * parameter:
        *   - inputs(integer) = the number of predictors in the dataset
        *   - nodes(integer) = the number of hidden nodes in the neural network.*/
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
        /*This is just the weight initialising function that randomizes the models initial weights
         * parameter:
         *   - inputs = the number of predictors in the dataset
         *   - nodes = the number of hidden nodes in the neural network.*/

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
        //showWeights();
    }

    public void showWeights(){
        /*This function display the current weight and bias values in the neural network,*/
        System.out.println("This is the new weights");
        for(double[] arrays: input_hiddenWeights){
            System.out.print(","+ Arrays.toString(arrays));
        }
        System.out.println(Arrays.toString(hidden_outputWeights));
        System.out.println(Arrays.toString(hiddenLayerBiases));
        System.out.println(outputBias);
    }

}
