import java.io.IOException;
import java.util.ArrayList;

public class BackProp_tanh extends BackPropagation{
    public BackProp_tanh(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    @Override
    public double activationFunction(double output) {
        /*this function changes the activation function of the previous back propagation algorithm uses tan function instead.
        * parameter:
        *   - output(double) =  the output/value to be activated. */
        output = (Math.exp(output)-Math.exp(-output))/(Math.exp(output)+Math.exp(-output));
        return output;
    }

    @Override
    public double derivedActivationFunction(double output) {
        /*this function derives the activation function of the previous back propagation algorithm uses tan function derivative instead.
         * parameter:
         *   - output(double) =  the output/value to be derived. */
        output = 1-(output*output);
        return output;
    }
}
