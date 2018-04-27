package model;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashSet;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Scanner;


public class DataSetsLoader {	
	ArrayList<Float> minValuesForFeatures			= new ArrayList<Float>();
	ArrayList<Float> maxValuesForFeatures			= new ArrayList<Float>();	
	public LinkedHashSet<String> dataSetClasses		= new LinkedHashSet<>();
	public BaseClassifier myclassifier;
	public String dataSetName;
	public HashSet<String> dataSetAttrsLabels		= new HashSet<>();
	public ArrayList<LabelledDataInstance> testDataSetList =new ArrayList<>();
	public ArrayList<LabelledDataInstance> trainingDataSetList =new ArrayList<>();
	
	@Override
	public String toString() {
		String strRep								= dataSetName + " [dataSetListSize=" + "]\n";
		strRep									   += "               [DataSetLabels= " + dataSetClasses + "]\n";
		strRep									   += "               [LabelsCount  = " + dataSetClasses.size() + "]\n";

		return strRep;
	}

	
	public void clear() {
		this.trainingDataSetList.clear();
		this.testDataSetList.clear();
		this.minValuesForFeatures.clear();
		this.maxValuesForFeatures.clear();
	}
	
	public void writeTestIrisDataSetForNN() {
		ArrayList<String> classes 					= new ArrayList<>(dataSetClasses);
//		Collections.shuffle(testDataSetList);
		List<String> lines 							=new ArrayList<String>();
		for (LabelledDataInstance instance:testDataSetList) {
			String line								="";
			for (String value :instance.featuresListAsStrings) {
				line								+=value+";";
			}
			line									+=classes.indexOf(instance.labelName);
			lines.add(line);
//			System.out.println("AddingLIne>>"+line);
		}
		Path file = Paths.get("irisDataTestSetNN.txt");
		try {
			Files.write(file, lines, Charset.forName("UTF-8"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}	
	
	
	public void writeIrisDataSetForNN() {
		ArrayList<String> classes 					= new ArrayList<>(dataSetClasses);
		Collections.shuffle(trainingDataSetList);
		List<String> lines 							=new ArrayList<String>();
		for (LabelledDataInstance instance:trainingDataSetList) {
			String line								="";
			for (String value :instance.featuresListAsStrings) {
				line								+=value+";";
			}
			line									+=classes.indexOf(instance.labelName);
			lines.add(line);
//			System.out.println("AddingLIne>>"+line);
		}
		Path file = Paths.get("irisDataNN.txt");
		try {
			Files.write(file, lines, Charset.forName("UTF-8"));
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}		
	}

	public ArrayList<Double> getValueFromOutputFile() {
		File fileObj 								= new File(System.getProperty("user.dir").replace('\\', '/') +"/IrisTestSetOut.txt");
		String line									="";
		ArrayList<Double> predictedOutputs			= new ArrayList<>();
		try (FileReader fileReader = new FileReader(fileObj);
				BufferedReader bufferedReader		= new BufferedReader(fileReader);){
				line								= bufferedReader.readLine();
				while (line!=null) {				 	
				 	predictedOutputs.add(Double.parseDouble(line));
				 	line							= bufferedReader.readLine();
				}
				
		} catch (IOException e) {
			System.out.println("FILE NOT FOUND !!");
		}		
		return predictedOutputs;
	}
	
	public String getValueFromTmpFile() {
		File fileObj 								= new File(System.getProperty("user.dir").replace('\\', '/') +"/IrisTestSetOut.txt");
		String line									="";
		try (FileReader fileReader = new FileReader(fileObj);
				BufferedReader bufferedReader		= new BufferedReader(fileReader);){
			 	line								= bufferedReader.readLine();
				
		} catch (IOException e) {
			System.out.println("FILE NOT FOUND !!");
		}		
		return line;
	}
	public String loadIrisDataSet(String filePath, ArrayList<LabelledDataInstance> dataSetList) {		
		
		File fileObj 								= new File(filePath);
		//Reading File Contents & Creating Java Representation Objects for DataSet For Classifier Classes//
		try (FileReader fileReader = new FileReader(fileObj);
				BufferedReader bufferedReader		= new BufferedReader(fileReader);){
			String line 							= bufferedReader.readLine();
			
			while (line != null) {
				String[] lineParts 					= line.split("  ");
				if (lineParts.length < 5) {
					break;
				}				

				List<String> lineList 				= Arrays.asList(lineParts);
				String featureLabel 				= lineList.get(lineList.size() - 1);
				List<String> featuresList 			= lineList.subList(0, 4);
				LabelledDataInstance dataInstance 	= new LabelledDataInstance(featuresList, featureLabel);
				dataSetClasses.add(featureLabel);
				dataInstance.parseInformationToValues(); // Converts Strings to Floats
				dataSetList.add(dataInstance);
				line 								= bufferedReader.readLine();
			}			
		} catch (IOException e) {
			System.out.println("FILE NOT FOUND !!");
		}
		
		String report								= toString();
		return report;
	}
	

}