package model;

import java.io.File;
import java.util.HashMap;
import org.joone.engine.FullSynapse;
import org.joone.engine.LinearLayer;
import org.joone.engine.Monitor;
import org.joone.engine.NeuralNetEvent;
import org.joone.engine.SigmoidLayer;
import org.joone.engine.SimpleLayer;
import org.joone.engine.learning.TeachingSynapse;
import org.joone.io.FileInputSynapse;
import org.joone.io.FileOutputSynapse;
import org.joone.io.MemoryInputSynapse;
import org.joone.net.NeuralNet;
import application.MainController;
import javafx.application.Platform;

public class Processor implements org.joone.engine.NeuralNetListener{
	private MainController uiWin ;
	private NeuralNet nnet 		= new NeuralNet();
	private double learningRate;
	private double momentum;
	private int epochs;
	public int epochsCounter	= 0;
	public int nodesPerLayerCount;
	public int hiddenLayers;
	HashMap<Integer, SimpleLayer> layersMap = new HashMap<>();	
	public Processor(MainController uiWin) {
		super();
		this.uiWin 				= uiWin;
		this.learningRate		= uiWin.getLearningRate();
		this.momentum			= uiWin.getMomentum();
		this.epochs				= uiWin.getEpochs();
	}
	
	public double checkOutputOfOneInstance(double [][] inputArray) {
		Monitor monitor	 		= nnet.getMonitor();
		int numberOfLayers 		= layersMap.keySet().size();
		double output			= -10.0;
		layersMap.get(0).removeAllInputs(); // FIRST LAYER IN NET IS ALWAYS THE INPUT
        MemoryInputSynapse  inputStream = new MemoryInputSynapse();
        inputStream.setInputArray(inputArray);
        inputStream.setAdvancedColumnSelector("1,2,3,4");
        layersMap.get(0).addInputSynapse(inputStream); 
        monitor.setTotCicles(1);
        monitor.setLearning(false);        
        monitor.setTrainingPatterns(1);
        FileOutputSynapse outputFile = new FileOutputSynapse();
        // set the output synapse to write the output of the net
        outputFile.setFileName(System.getProperty("user.dir").replace('\\', '/') +"/tmp.txt");
        // inject the input and get the output        
        if(nnet!=null) {
        	nnet.addOutputSynapse(outputFile);
//            System.out.println("Testing Performance"+nnet.check());
            nnet.start();
            monitor.Go();
            nnet.join();
//            System.out.println(layersMap.get(numberOfLayers-1).getLastOutputs().length);
    		output 				= layersMap.get(numberOfLayers-1).getLastOutputs()[0];
    		
    		nnet.removeOutputSynapse(outputFile);
        }
		
		
        return output;    
	}
	public void  measurePerformance() {
		NeuralNet newNet		= nnet.cloneNet();
		Monitor monitor	 		= newNet.getMonitor();
		
		newNet.getInputLayer().removeAllInputs(); // FIRST LAYER IN NET IS ALWAYS THE INPUT
		FileInputSynapse inputStream = new FileInputSynapse();
		/* The first 4 columns contain the input values */
		inputStream.setAdvancedColumnSelector("1,2,3,4");
		/* This is the file that contains the input data */
		String trainingFilePath	= System.getProperty("user.dir").replace('\\', '/') + "/irisDataTestSetNN.txt";
		inputStream.setInputFile(new File(trainingFilePath));		
		newNet.getInputLayer().addInputSynapse(inputStream); // FIRST LAYER IN NET IS ALWAYS THE INPUT	
		
        monitor.setTotCicles(1);
        monitor.setLearning(false);
        monitor.setTrainingPatterns(uiWin.myDataLoader.testDataSetList.size());
        FileOutputSynapse output=new FileOutputSynapse();
        // set the output synapse to write the output of the net
        output.setFileName(System.getProperty("user.dir").replace('\\', '/') +"/IrisTestSetOut.txt");
        // inject the input and get the output
        if(newNet!=null) {
        	newNet.addOutputSynapse(output);
//            System.out.println("Testing Performance"+newNet.check());
            newNet.start();
            monitor.Go();
            newNet.join();
        }
        
		
	}
	public void buildNNTest2(int numberOfHiddenLayers ,int numberOfNeuronsPerHLayer) {
		nnet.stop();
		nnet.removeAllInputs();
		nnet.removeAllListeners();
		nnet.removeAllOutputs();
		nnet = new NeuralNet();
		int layersCount				= 0;
		nodesPerLayerCount			= numberOfNeuronsPerHLayer;	
		hiddenLayers				= numberOfHiddenLayers;
		LinearLayer input 			= new LinearLayer("INPUT");
		input.setRows(4);
		layersMap.clear();
		layersMap.put(layersCount,input);
		for (int i=0;i<numberOfHiddenLayers;i++) {
			layersCount++;
			SigmoidLayer hidden = new SigmoidLayer("Hidden-"+i);
			hidden.setRows(numberOfNeuronsPerHLayer);
			layersMap.put(layersCount, hidden);
		}
		
		LinearLayer output = new LinearLayer("OUTPUT");
		layersCount++;
		output.setRows(1);
		layersMap.put(layersCount,output);
		
		//Connecting Layers with Synapses
		for(int i =0; i<layersCount;i++) {
			FullSynapse synapse_IH = new FullSynapse(); /* Layer i < Layer i+1*/
			layersMap.get(i).addOutputSynapse(synapse_IH);
			layersMap.get(i+1).addInputSynapse(synapse_IH);
		}
		
		System.out.println(layersMap);
		//Adding Layers To NN
		
		nnet.addLayer(input, NeuralNet.INPUT_LAYER);
		for (int i=1;i<layersCount;i++) {
			nnet.addLayer(layersMap.get(i),NeuralNet.HIDDEN_LAYER);
			System.out.println("Adding Layer"+i);
		}
		nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);	
		
		Monitor monitor = nnet.getMonitor();
		monitor.setLearningRate(0.7);
		monitor.setMomentum(0.5);
		
		monitor.addNeuralNetListener(this);		
		
		FileInputSynapse inputStream = new FileInputSynapse();
		/* The first 4 columns contain the input values */
		inputStream.setAdvancedColumnSelector("1,2,3,4");
		/* This is the file that contains the input data */
		String trainingFilePath		= System.getProperty("user.dir").replace('\\', '/') + "/irisDataNN.txt";
		inputStream.setInputFile(new File(trainingFilePath));		
		
		input.addInputSynapse(inputStream);
		
		TeachingSynapse trainer = new TeachingSynapse();
		/* Setting of the file containing the desired responses, provided by a FileInputSynapse */
		FileInputSynapse samples = new FileInputSynapse();
		
		samples.setInputFile(new File(trainingFilePath));
		trainer.setDesired(samples);
		/* The output values are on the Fifth column of the file */
		samples.setAdvancedColumnSelector("5");
		/* We add it to the neural network */
		nnet.setTeacher(trainer);
		
		output.addOutputSynapse(trainer);
		monitor.setTrainingPatterns(uiWin.myDataLoader.trainingDataSetList.size()); /* # of rows contained in the input file */
		monitor.setTotCicles(10000); /* How many times the net must be trained on the input patterns */
		monitor.setLearning(true); /* The net must be trained */
//		System.out.println(monitor.getGlobalError());
//		nnet.go(); /* The network starts the training job */
//		System.out.println(monitor.getGlobalError());
		
	}
	public void buildNNTest() {
		LinearLayer input = new LinearLayer();
		SigmoidLayer hidden = new SigmoidLayer();
		SigmoidLayer output = new SigmoidLayer();
		
		input.setRows(4);
		hidden.setRows(10);
		output.setRows(1);
		
		FullSynapse synapse_IH = new FullSynapse(); /* Input  -> Hidden conn. */
		FullSynapse synapse_HO = new FullSynapse(); /* Hidden -> Output conn. */
		
		input.addOutputSynapse(synapse_IH);
		hidden.addInputSynapse(synapse_IH);
		hidden.addOutputSynapse(synapse_HO);
		output.addInputSynapse(synapse_HO);
		
		nnet = new NeuralNet();
		nnet.addLayer(input, NeuralNet.INPUT_LAYER);
		nnet.addLayer(hidden, NeuralNet.HIDDEN_LAYER);
		nnet.addLayer(output, NeuralNet.OUTPUT_LAYER);	
		
		Monitor monitor = nnet.getMonitor();
		monitor.setLearningRate(0.7);
		monitor.setMomentum(0.5);
		
		monitor.addNeuralNetListener(this);		
		
		FileInputSynapse inputStream = new FileInputSynapse();
		/* The first 4 columns contain the input values */
		inputStream.setAdvancedColumnSelector("1,2,3,4");
		/* This is the file that contains the input data */
		String trainingFilePath		= System.getProperty("user.dir").replace('\\', '/') + "/irisDataNN.txt";
		inputStream.setInputFile(new File(trainingFilePath));		
		
		input.addInputSynapse(inputStream);
		
		TeachingSynapse trainer = new TeachingSynapse();
		/* Setting of the file containing the desired responses, provided by a FileInputSynapse */
		FileInputSynapse samples = new FileInputSynapse();
		
		samples.setInputFile(new File(trainingFilePath));
		trainer.setDesired(samples);
		/* The output values are on the Fifth column of the file */
		samples.setAdvancedColumnSelector("5");
		/* We add it to the neural network */
		nnet.setTeacher(trainer);
		
		output.addOutputSynapse(trainer);
		monitor.setTrainingPatterns(75); /* # of rows contained in the input file */
		monitor.setTotCicles(10000); /* How many times the net must be trained on the input patterns */
		monitor.setLearning(true); /* The net must be trained */
//		System.out.println(monitor.getGlobalError());
//		nnet.go(); /* The network starts the training job */
//		System.out.println(monitor.getGlobalError());
		
	}
	public void run() {
		this.learningRate		= uiWin.getLearningRate();
		this.momentum			= uiWin.getMomentum();
		this.epochs				= uiWin.getEpochs();	
		Monitor monitor 		= nnet.getMonitor();
		monitor.setLearningRate(learningRate);
		monitor.setMomentum(momentum);		
		monitor.setTotCicles(epochs);	
		nnet.go(); /* The network starts the training job */
	}
	@Override
	public void netStarted(NeuralNetEvent e) {
		System.out.println("NetStarted");
		
	}

	@Override
	public void cicleTerminated(NeuralNetEvent e) {
		epochsCounter++;
		Monitor m = (Monitor)e.getSource();
		String message = "Momentum="+m.getMomentum()+"\nLearning Rate = "+m.getLearningRate()+"\nGlobal Error = "+m.getGlobalError()+"\n"
						+"Current Epoch ="+epochsCounter+"\nLayersInNetwork="+layersMap+"\nNodesPerHiddenLayer="
						+nodesPerLayerCount;
		Platform.runLater(()->uiWin.appendToStatusText(message));
		Platform.runLater(()->uiWin.learningcurve.put(epochsCounter, m.getGlobalError()));
		
		Platform.runLater(()->uiWin.drawLearningCurveUI());
		if(epochsCounter%10==0) {
			Platform.runLater(()->uiWin.measurePerformance());
		}
		
		
//		uiWin.appendToStatusText(m.getGlobalError()+","+m.getCurrentCicle());
//		System.out.println(m.getGlobalError()+","+m.getCurrentCicle());
		
	}

	@Override
	public void netStopped(NeuralNetEvent e) {
		
		Monitor m = (Monitor)e.getSource();
		String message = "Momentum="+m.getMomentum()+" Learning Rate = "+m.getLearningRate()+" Global Error = "+m.getGlobalError()+" "
						+"Current Epoch ="+epochsCounter+" HiddenLayers ="+hiddenLayers+" NodesPerHiddenLayer="
						+nodesPerLayerCount;
		Platform.runLater(()->uiWin.drawLearningCurveUI());
		Platform.runLater(()->uiWin.updateLearningCurveChart(message));
		
	}

	@Override
	public void errorChanged(NeuralNetEvent e) {
		
//		Monitor m = (Monitor)e.getSource();
//		int numberOfLayers = layersMap.keySet().size();
//		System.out.println(layersMap.get(numberOfLayers-1).getLastOutputs()[0]);
		
	}

	@Override
	public void netStoppedError(NeuralNetEvent e, String error) {
		// TODO Auto-generated method stub
		System.out.println("NETERROR!!\n"+error);
	}
}
