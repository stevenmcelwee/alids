package com.stevenmcelwee.ActiveLearnerIDS;

import weka.core.Instances;
import weka.core.Instance;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.ObjectInputStream;
import java.io.ObjectOutputStream;
import weka.classifiers.trees.RandomForest;


public class Learner implements java.io.Serializable {
	private static final long serialVersionUID = 2486112232552746137L;  // unique id for serialization

	private RandomForest learner;
	private int numTrees = 100; // Default will be 100

	public void train(Instances trainingDataset) throws Exception {
		trainingDataset.setClassIndex(trainingDataset.numAttributes()-1);
		this.learner = new RandomForest();
		this.learner.setNumTrees(this.numTrees);
		this.learner.buildClassifier(trainingDataset);
	}
	
	
	public double classifyInstance(Instance inst) throws Exception {
		return this.learner.classifyInstance(inst);
	}
	
	public double[] distributionForInstance(Instance inst) throws Exception {
		return this.learner.distributionForInstance(inst);
	}
	
	public Instances classify(Instances unlabeledDataset) throws Exception {
		unlabeledDataset.setClassIndex(unlabeledDataset.numAttributes()-1);
		
		// Make a copy of the dataset
		Instances labeledDataset = new Instances(unlabeledDataset);
		
		for (int i = 0; i < unlabeledDataset.numInstances(); i++) {
			unlabeledDataset.instance(i).setClassMissing(); // Make sure label is clear before we classify
			double label = this.learner.classifyInstance(unlabeledDataset.instance(i));
			labeledDataset.instance(i).setClassValue(label);
		}
		return labeledDataset;
	}
	
	public void save(String fileName) throws IOException {
        ObjectOutputStream out = new ObjectOutputStream(new FileOutputStream(fileName));
        out.writeObject(this);
        out.close();
	}
	
	public static Learner load(String fileName) throws IOException, ClassNotFoundException {
	      Learner learner = null;
	      ObjectInputStream in = new ObjectInputStream(new FileInputStream(fileName));
	      learner = (Learner) in.readObject();
	      in.close();
	      return learner;
	}

	// Getters and setters
	public int getNumTrees() {
		return numTrees;
	}

	public void setNumTrees(int numTrees) {
		this.numTrees = numTrees;
	}
	
	
	
}
