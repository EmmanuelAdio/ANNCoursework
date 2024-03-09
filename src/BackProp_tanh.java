import java.io.IOException;
import java.util.ArrayList;

public class BackProp_tanh extends BackPropagation{
    public BackProp_tanh(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) throws IOException {
        super(dataset, valDataset, nodes, epochs);
    }

    @Override
    public double activationFunction(double output) {
        output = (Math.exp(output)-Math.exp(-output))/(Math.exp(output)+Math.exp(-output));
        return output;
    }

    @Override
    public double derivedActivationFunction(double output) {
        output = 1-(output*output);
        return output;
    }
}
