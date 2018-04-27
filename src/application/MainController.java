package application;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

import javax.imageio.ImageIO;

import javafx.embed.swing.SwingFXUtils;
import javafx.embed.swing.SwingNode;
import javafx.fxml.FXML;
import javafx.geometry.Bounds;
import javafx.scene.Group;
import javafx.scene.SnapshotParameters;
import javafx.scene.canvas.Canvas;
import javafx.scene.canvas.GraphicsContext;
import javafx.scene.chart.LineChart;
import javafx.scene.chart.NumberAxis;
import javafx.scene.chart.XYChart;
import javafx.scene.control.ChoiceDialog;
import javafx.scene.control.Label;
import javafx.scene.control.ScrollPane;
import javafx.scene.control.Slider;
import javafx.scene.control.TextArea;
import javafx.scene.control.TextField;
import javafx.scene.control.TextInputDialog;
import javafx.scene.image.ImageView;
import javafx.scene.image.WritableImage;
import javafx.scene.layout.HBox;
import javafx.scene.layout.Priority;
import javafx.scene.layout.VBox;

import model.DataSetsLoader;
import model.LabelledDataInstance;
import model.Processor;


public class MainController {
	// UI ELEMENTS
	@FXML TextArea StatusTA;
	@FXML Label MainLabel;		
	@FXML Label StatusLB;
	@FXML HBox PreviewHB;
	@FXML TextField HiddenLayersCountTF;
	@FXML TextField NeurounsCountPerHlayerTF;
	@FXML TextField EpochsTF;
	@FXML TextField LearningRateTF;
	@FXML VBox LearningCurveBox;
	@FXML TextField MomentumTF;
	@FXML VBox SnapShotPreviewVB;
	@FXML Label AccuracyLB;
	private Main main;
    private NumberAxis xAxis 					= new NumberAxis();
    private NumberAxis yAxis 					= new NumberAxis();
    private LineChart<Number,Number> lineChart  = new LineChart<Number,Number>(xAxis,yAxis);
    
	public DataSetsLoader myDataLoader = new DataSetsLoader();
	private int totalEpochsCount				= 0;
	public LinkedHashMap<Integer, Double> learningcurve=new LinkedHashMap<>();
	public LinkedHashMap<Integer, Double> ValidationCurve=new LinkedHashMap<>();
	private Processor myModel;			
	
	public void setMain(Main main) {
		this.main = main;

//        loadDataSet();
		myModel							= new Processor(this);
		setupLearningCurveChart();
		LearningCurveBox.getChildren().add(lineChart);
		LearningCurveBox.setVgrow(lineChart, Priority.ALWAYS);
		myDataLoader.loadIrisDataSet(System.getProperty("user.dir").replace('\\', '/') + "/iris-training.txt",myDataLoader.trainingDataSetList);
		myDataLoader.loadIrisDataSet(System.getProperty("user.dir").replace('\\', '/') + "/iris-test.txt",myDataLoader.testDataSetList);
		myDataLoader.writeIrisDataSetForNN();
		myDataLoader.writeTestIrisDataSetForNN();
        myModel.buildNNTest();  
        
        
	}
	public void buildNNFromUiAttrs() {
		myModel.epochsCounter=0;
		learningcurve.clear();
		ValidationCurve.clear();
		myModel.buildNNTest2(Integer.parseInt(HiddenLayersCountTF.getText()),Integer.parseInt(NeurounsCountPerHlayerTF.getText()));
		appendToStatusText("Biult New NN\n HiddenLayers="+HiddenLayersCountTF.getText()
		+"\nNeuronsPerHiddenLayer="+NeurounsCountPerHlayerTF.getText());
	}
	
	public void resampleLearningCurve() {
		TextInputDialog dialog = new TextInputDialog("0.01");
		dialog.setTitle("Resample Learning Curve");
		dialog.setHeaderText("fraction of points to Skip");
		dialog.setContentText("double fraction = ");

		// Traditional way to get the response value.
		Optional<String> result = dialog.showAndWait();
		if (result.isPresent()){
			drawLearningCurve(Double.parseDouble(result.get()));
		}		
		
	}	
	public void drawLearningCurveUI(){
		drawLearningCurve(0.01);
	}	
	
	public void updateLearningCurveChart(String Message) {
		StatusLB.setText(Message);
	}
	public void drawLearningCurve(double fractionOfPointsToKeep) {
		lineChart.getData().clear();
		Map.Entry<Integer,Double> firstEntry	= (Map.Entry<Integer,Double>) learningcurve.entrySet().toArray()[0];		
		XYChart.Series<Number,Number> series 	= new XYChart.Series<>();
		series.setName("Error");
		series.getData().add(new XYChart.Data(firstEntry.getKey(), firstEntry.getValue()));
		double pointsToSkip						= learningcurve.size()*fractionOfPointsToKeep;
		Iterator iter							=learningcurve.entrySet().iterator();
		while(iter.hasNext()) {
			Map.Entry<Integer,Double> point		= (Map.Entry)iter.next();
			series.getData().add(new XYChart.Data(point.getKey(), point.getValue()));
			for(int i=0; i<(int)pointsToSkip;i++) {
				if(iter.hasNext()) {iter.next();}
			}
		}
		
		Map.Entry<Integer,Double> lastEntry					= (Map.Entry<Integer,Double>) learningcurve.entrySet().toArray()[0];		
		series.getData().add(new XYChart.Data(lastEntry.getKey(), lastEntry.getValue()));

		lineChart.getData().add(series);
		drawValidationCurve();
	}	
	public void drawValidationCurve() {
		XYChart.Series<Number,Number> series 	= new XYChart.Series<>();
		series.setName("TestDataSet Error");
		series.getData().clear();
		Iterator iter							=ValidationCurve.entrySet().iterator();
		while(iter.hasNext()) {
			Map.Entry<Integer,Double> point		= (Map.Entry)iter.next();
			series.getData().add(new XYChart.Data(point.getKey(), point.getValue()));
		}
		lineChart.getData().add(series);
		
	}
		
		

	public void setupLearningCurveChart() {
        xAxis.setLabel("Epoch");
        yAxis.setLabel("Error");        
//        lineChart.setTitle("Learning Curve");
        lineChart.setCreateSymbols(false);
        lineChart.setAnimated(false);
        lineChart.getXAxis().setAutoRanging(true);
        lineChart.getXAxis();
		lineChart.getYAxis().setAutoRanging(true);
		lineChart.setVerticalGridLinesVisible(true);   
		

	}
	
	public double getLearningRate() {
		return Double.parseDouble(LearningRateTF.getText());
	}
	
	public double getMomentum() {
		return Double.parseDouble(MomentumTF.getText());
	}
	public int getEpochs() {
		return Integer.parseInt(EpochsTF.getText());
	}
	public void runNN() {
		myModel.run();
		
	}
	public void appendToStatusText(String message) {
		StatusTA.setText(message+"\n");
	}
	public void loadDataSet() {
		// Loads Images From Disk 
		totalEpochsCount			= 0;
		learningcurve.clear();
//		System.out.println("Loading Images DataSet");
		String trainingFilePath		= System.getProperty("user.dir").replace('\\', '/') + "/image.data";
		myDataLoader.clear();
		myDataLoader.dataSetName	= "ImageDataSet";
		StatusTA.setText(myDataLoader.toString());

	
	}
	private String promptUserForChoice(List<String> dialogData, String message) {
		ChoiceDialog<String> dialog = new ChoiceDialog<String>(dialogData.get(0), dialogData);
		dialog.setTitle("");
		dialog.setHeaderText(message);
		Optional<String> result = dialog.showAndWait();
		String selected = "cancelled.";

		if (result.isPresent()) {

			selected = result.get();
		}

		return selected;
	}
	public void measurePerformance2() {
		
		ArrayList<String> classes 				= new ArrayList<>(myDataLoader.dataSetClasses);
		int correctCount						= 0;
		for (LabelledDataInstance instance:myDataLoader.testDataSetList) {
			double [][] inputArray					= {
					{Double.parseDouble(instance.featuresListAsStrings.get(0)),
					Double.parseDouble(instance.featuresListAsStrings.get(1)),
					Double.parseDouble(instance.featuresListAsStrings.get(2)),
					Double.parseDouble(instance.featuresListAsStrings.get(3)),
					}
			};
			double output							= myModel.checkOutputOfOneInstance(inputArray);
			int groundTruth							= classes.indexOf(instance.labelName);
			double roundedOutput					= Double.parseDouble(myDataLoader.getValueFromTmpFile());
//			System.out.println(output+"<<>>"+groundTruth+"<<>>"+roundedOutput);
			if ((int)roundedOutput == groundTruth) {
				correctCount++;
			}
			double accuracy							= ((double)correctCount /(double)myDataLoader.trainingDataSetList.size())*100.0;
			AccuracyLB.setText("CorrectCount = " + correctCount +
							" Accuracy   = " + accuracy);
		}
	}
	public void measurePerformance() {
		myModel.measurePerformance();
		ArrayList<Double> outputPredictions		= myDataLoader.getValueFromOutputFile();
		ArrayList<String> classes 				= new ArrayList<>(myDataLoader.dataSetClasses);
		int correctCount						= 0;
		
		for (LabelledDataInstance instance :myDataLoader.testDataSetList) {
			int groundTruth						= classes.indexOf(instance.labelName);
			int index							= myDataLoader.testDataSetList.indexOf(instance);
			double prediction					= outputPredictions.get(index);
//			System.out.println("Index :"+index+" Target "+ groundTruth+" Prediction "+ Math.round(prediction)+ " "+prediction);
			if (Math.round(prediction)==groundTruth) {
				correctCount++;
			};
		}
		double accuracy							= ((double)correctCount /(double)myDataLoader.trainingDataSetList.size())*100.0;
		double error							= 1.0 -((double)correctCount /(double)myDataLoader.trainingDataSetList.size());
		ValidationCurve.put(myModel.epochsCounter, error);
		AccuracyLB.setText("CorrectCount = " + correctCount +
						" Accuracy   = " + accuracy);
		
		
	}
	public void saveAsPng(String fileName) {
		double scale							= 5;
		Bounds bounds 							= LearningCurveBox.getLayoutBounds();
		WritableImage image 					= new WritableImage(
	            (int) Math.round(bounds.getWidth() * scale),
	            (int) Math.round(bounds.getHeight() * scale));
		
		SnapshotParameters snapshotParams   	= new SnapshotParameters();
		snapshotParams.setFill(javafx.scene.paint.Color.rgb(40, 40, 40, 1));
		snapshotParams.setTransform(javafx.scene.transform.Transform.scale(scale, scale));
		
//	    WritableImage image2 					= TreeP.snapshot(snapshotParams,null);
    	
	    ImageView view 							= new ImageView(LearningCurveBox.snapshot(snapshotParams, image));
	    File file = new File(fileName+".png");
	    
	    try {
	        ImageIO.write(SwingFXUtils.fromFXImage(view.getImage(), null), "png", file);
	    } catch (IOException e) {
	        
	    }
	}	
	public void saveSnapShot() {
		saveAsPng("LR_" + LearningRateTF.getText()+"_EPC_"+myModel.epochsCounter);
	}
		
}
