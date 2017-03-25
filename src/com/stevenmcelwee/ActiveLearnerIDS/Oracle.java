package com.stevenmcelwee.ActiveLearnerIDS;

import weka.core.Instances;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.Evaluation;
import java.util.Random;
import java.io.IOException;
import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import java.io.FileInputStream;
import java.io.ObjectInputStream;


public class Oracle implements java.io.Serializable {
	
	private static final long serialVersionUID = 2426162292522741137L;  // unique id for serialization
	private RandomForest learner;

	public Instances query(Instances unlabeledDataset) throws Exception {
		
		unlabeledDataset.setClassIndex(unlabeledDataset.numAttributes()-1);
		
		Instances labeledDataset = new Instances(unlabeledDataset);
		
		for (int i = 0; i < unlabeledDataset.numInstances(); i++) {
			double label = this.learner.classifyInstance(unlabeledDataset.instance(i));
			labeledDataset.instance(i).setClassValue(label);
		}
		return labeledDataset;
	}
	
	public String train(Instances labeledDataset) throws Exception {
		// Variable that will be returned with results and status
		String resultsMessage = "";
		
		// TODO: Consider deduping the data before training
		
		labeledDataset.setClassIndex(labeledDataset.numAttributes()-1);
		this.learner = new RandomForest();
		this.learner.setNumTrees(100);
		this.learner.buildClassifier(labeledDataset);
		
		int numFolds = 2;
		
	    Evaluation evaluation = new Evaluation(labeledDataset);
	    evaluation.crossValidateModel(this.learner, labeledDataset, numFolds, new Random(1));
	    resultsMessage += evaluation.toSummaryString("\nResults\n======\n", true) + "\n";
	    resultsMessage += evaluation.toClassDetailsString() + "\n";
	    resultsMessage += "Results For Class -1- " + "\n";
	    resultsMessage += "Precision=  " + evaluation.precision(0) + "\n";
	    resultsMessage += "Recall=  " + evaluation.recall(0) + "\n";
	    resultsMessage += "F-measure=  " + evaluation.fMeasure(0) + "\n";
	    resultsMessage += "Results For Class -2- " + "\n";
	    resultsMessage += "Precision=  " + evaluation.precision(1) + "\n";
	    resultsMessage += "Recall=  " + evaluation.recall(1) + "\n";
	    resultsMessage += "F-measure=  " + evaluation.fMeasure(1) + "\n";
	    return resultsMessage;
	    
	}
	
	public void save(String fileName) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
        out.writeObject(this);
        out.close();
	}
	
	public static Oracle load(String fileName) throws IOException, ClassNotFoundException {
	      Oracle oracle = null;
	      ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
	      oracle = (Oracle) in.readObject();
	      in.close();
	      return oracle;
	}
	
}
