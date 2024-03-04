import java.util.ArrayList;

public class BoldDriver extends BackPropagation {

    public BoldDriver(ArrayList<ArrayList<Double>> dataset, ArrayList<ArrayList<Double>> valDataset, int nodes, int epochs) {
        super(dataset, valDataset, nodes, epochs);
    }
}
