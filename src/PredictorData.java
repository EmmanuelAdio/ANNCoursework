import java.io.File;  // Import the File class
import java.io.FileNotFoundException;  // Import this class to handle errors
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Scanner; // Import the Scanner class to read text files

public class PredictorData {
    /*This is the Class just makes it easier for multiple datasets to be collected and used in the program.
    * Dataset (ArrayList<ArrayList<Double>>) = This is the data set in a 2D array format so it cna be used within the program.*/
    private ArrayList<ArrayList<Double>> Dataset;
    public PredictorData(String file) throws FileNotFoundException {
        setDataset(file);
    }

    public void setDataset(String dataset) throws FileNotFoundException {
        /*This is the method that collects the information in the dataset file and puts it in the correct 2d array format to be used in the program.
        * dataset (String) = The directory of the dataset file.
        * */
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
        /*This is the get method for the dataset collected and put into the 2d array format for the program.
        * */
        return Dataset;
    }
}
