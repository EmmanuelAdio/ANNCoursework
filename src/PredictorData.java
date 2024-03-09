import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner; // Import the Scanner class to read text files

public class PredictorData {
    /*This function just makes it easier for datasets to be collected and used in the program.*/
    private ArrayList<ArrayList<Double>> Dataset;
    public PredictorData(String file) throws FileNotFoundException {
        setDataset(file);
    }

    public void setDataset(String dataset) throws FileNotFoundException {
        ArrayList<ArrayList<Double>> Data = new ArrayList<ArrayList<Double>>();
        File DatasetFile = new File(dataset);
        Scanner scan = new Scanner(DatasetFile);
        while(scan.hasNextLine()){
            ArrayList<Double> dataSample = new ArrayList<Double>();
            String[] dataSampleRaw = scan.nextLine().split("\\|");

            for(String val : dataSampleRaw){
                dataSample.add(Double.parseDouble(val.replace("\uFEFF","")));
            }

            Data.add(dataSample);
        }

        Dataset = Data;
    }

    public ArrayList<ArrayList<Double>> getDataset() {
        return Dataset;
    }
}
