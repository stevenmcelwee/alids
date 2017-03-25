package com.stevenmcelwee.ActiveLearnerIDS;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ArffLoader.ArffReader;
import weka.clusterers.SimpleKMeans;
import weka.core.AttributeStats;
import weka.core.Instance;
import weka.filters.Filter;
import weka.filters.unsupervised.instance.RemoveDuplicates;
import weka.filters.unsupervised.instance.Resample;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.Random;
import java.util.Properties;

import com.stevenmcelwee.ActiveLearnerIDS.Oracle;
import com.stevenmcelwee.ActiveLearnerIDS.Learner;
import com.stevenmcelwee.ActiveLearnerIDS.BatchResult;


public class ActiveLearnerIDS {

	private Learner   learner;
	private Oracle    oracle;
	private Instances masterTrainingDataset; 
	private Instances unlabeledDataset; 
	private Instances classifiedDataset; // The unlabeled dataset after processing by the classifier
	private Instances candidateDataset;  // The candidates for sending to the oracle

	// Settings to be loaded from properties file
	private static String unlabeledDatasetFileName;
	private static String oracleTrainingDatasetFileName;
	private static String oracleFileName;
	private static String learnerFileName;
	private static String masterDataFileName;
	private static int    numTrees;
	private static double minConfidence;  // Threshold for accepting a classification result
	private static int    batchSize;
	private static String strategy;
	private static double numInstancesForOracle; // This must be a double to deal with precision in division
	private static boolean saveLearner = false; // False by default. Can be overridden in prop file
	
	public static void main(String[] args) {
		
		// Check that an argument was supplied for the properties file
		if (args.length < 1) {
			System.out.println("A filename for the property file must be provided as the first argument.");
			System.exit(1);
		}
		// Load the properties file
		try {
			BufferedReader reader = new BufferedReader(new FileReader(args[0]));
			Properties prop = new Properties();
			prop.load(reader);
			unlabeledDatasetFileName = prop.getProperty("unlabeledDatasetFileName");
			oracleTrainingDatasetFileName = prop.getProperty("oracleTrainingDatasetFileName");
			oracleFileName = prop.getProperty("oracleFileName");
			learnerFileName = prop.getProperty("learnerFileName");
			masterDataFileName = prop.getProperty("masterDataFileName");
			numTrees = new Integer(prop.getProperty("numTrees"));
			minConfidence = new Double(prop.getProperty("minConfidence"));
			batchSize = new Integer(prop.getProperty("batchSize"));
			strategy = prop.getProperty("strategy");
			numInstancesForOracle = new Double(prop.getProperty("numInstancesForOracle"));
			saveLearner = new Boolean(prop.getProperty("saveLearner"));
		}
		catch (IOException e) {
			System.out.println("Error: Unable to load properties file.");
			System.exit(1);
		}

		// Instantiate this class
		ActiveLearnerIDS alids = new ActiveLearnerIDS();
		
		try {
			
			// Open reader for unlabled dataset
			BufferedReader reader = new BufferedReader(new FileReader(unlabeledDatasetFileName));
			ArffReader arff = new ArffReader(reader, 10000);
			Instances data = arff.getStructure();
			data.setClassIndex(data.numAttributes() - 1);
			Instance inst;
			int numBatches = 1;
			boolean haveData = true;

			BatchResult result = new BatchResult();
			
			// Print out the results header
			System.out.println(result.getHeader());
			
			while (haveData) {
				// Clear out dataset before each loop
				data.delete();
				// Instantiate batch result object for data collection
				result = new BatchResult(numBatches, batchSize);
				// Create batches
				int batchCounter = 0;
				while (batchCounter < batchSize) {
					if ((inst = arff.readInstance(data)) != null) {
						data.add(inst);
					}
					else {
						haveData = false;
						break;
					}
					batchCounter++;
				}
				// Run the process on the batched dataset
				
				result = alids.go(data, result);
				System.out.println(result.toString());
				System.gc(); // Clear out memory before next run; may not be needed
			    numBatches++; // Increment counter for batch number
			    
			}
			// File processed - save the final master dataset
			alids.saveDatafile(masterDataFileName, alids.getMasterTrainingDataset()); 
			// Save learner for reuse if this options is set
			if (saveLearner) {
				alids.saveLearner(learnerFileName);
			}
		}
		catch (Exception e) {
			System.out.println("Error: " + e);
			e.printStackTrace();
		}
	}
		
	private BatchResult go(Instances unlabeledDataset, BatchResult result) throws Exception {
			
		// Take care of some settings for recording in the results
		result.setMinConfidence(minConfidence);
		result.setMaxOracleQuerySize(numInstancesForOracle);
		result.setStrategy(strategy);
		
		// Does the oracle exist on the file system?
		if (this.oracle != null) {
			// Oracle already loaded. Do nothing.
		}
		else {
			// Create or load the oracle
			try {
				this.oracle = Oracle.load(oracleFileName);
			}
			catch(Exception e) {
				// 	Unable to retrieve the oracle - build a new one
				this.buildOracle();
			}
		}
		
		// Assign unlabeled dataset to the the global variable
		this.unlabeledDataset = new Instances(unlabeledDataset);			
		this.unlabeledDataset.setClassIndex(this.unlabeledDataset.numAttributes()-1);
		
		
		// Clear out existing class values - mostly for testing, but might keep it anyway
		for (int i = 0; i < this.unlabeledDataset.numInstances(); i++) {
			this.unlabeledDataset.instance(i).setClassMissing();
		}
	
		boolean learnerExists = false;
		if (this.learner != null) {
			// Do nothing. It's already loaded
			learnerExists = true;
		}
		else {
			// Does the learner exist on the file system?
			try {
				this.learner = Learner.load(learnerFileName);
				learnerExists = true;
			}
			catch (IOException e){
				// Learner file not found
			}
			catch (ClassNotFoundException e) {
				// Corrupted learner file
			}
		}
		
		// Instantiate the masterdata dataset 
		if (this.masterTrainingDataset == null) {
			this.masterTrainingDataset = new Instances(this.unlabeledDataset,0);
			this.masterTrainingDataset.setClassIndex(this.masterTrainingDataset.numAttributes()-1);
		}
		
		if (learnerExists) {
			// Instantiate candidate dataset as empty set for collecting low confidence/missing class results
			this.candidateDataset = new Instances(this.unlabeledDataset,0);
			this.candidateDataset.setClassIndex(this.candidateDataset.numAttributes()-1);

			// Make copy of unlabeled dataset for classification
			this.classifiedDataset = new Instances(this.unlabeledDataset);
			this.classifiedDataset.setClassIndex(this.classifiedDataset.numAttributes()-1);
			
			// Classify unlabeled dataset
			int tallyOfNewCandidates = 0;
			for (int i = 0; i < this.unlabeledDataset.numInstances(); i++) {
				
				double label = this.learner.classifyInstance(unlabeledDataset.instance(i));
				this.classifiedDataset.instance(i).setClassValue(label);
				// 	Evaluate distribution results to see if the confidence is high enough to add to master dataset
				double[] prediction = this.learner.distributionForInstance(this.classifiedDataset.instance(i));
				double maxPrediction = 0;
				for (int j = 0; j < prediction.length; j++) {
					if (prediction[j] > maxPrediction) maxPrediction = prediction[j];
				}
				if (this.classifiedDataset.instance(i).classIsMissing()) {
					// Could not classify dataset, so add it to the candidate list for sending to oracle
					this.candidateDataset.add(this.classifiedDataset.instance(i));
				}
				else if (maxPrediction < minConfidence) {
					this.candidateDataset.add(this.classifiedDataset.instance(i));
				}
				else {
					// Instance classified correctly and has high confidence. Add it to the master training dataset
					this.masterTrainingDataset.add(this.classifiedDataset.instance(i));
					tallyOfNewCandidates++;
				}
			}
			if (tallyOfNewCandidates == 0) {
				this.candidateDataset.addAll(this.classifiedDataset);
			}
		}
		else {
			// Unable to retrieve the learner - build a new one
			this.learner = new Learner();
			this.learner.setNumTrees(numTrees);
			this.candidateDataset = new Instances(this.unlabeledDataset); // copy all unlabeled data into candidate set
		}
		
		// Add number of candidates to the results object
		result.setNumLowConfidenceResults(this.candidateDataset.numInstances());

		// Start loop of query/learn
		// With classified or new data, create a query dataset to send to the oracle
		Instances queryDataset = this.createOracleQueryDataset();
		result.setActualOracleQuerySize(queryDataset.numInstances());
		
		// Query the oracle with the query dataset
		Instances oracleAnswersDataset = this.oracle.query(queryDataset);

		// Add oracle results to master dataset for training
		this.masterTrainingDataset.addAll(oracleAnswersDataset);
				
		// Deduplicate the master training dataset to speed up training the learner
		RemoveDuplicates remove = new RemoveDuplicates();
		remove.setInputFormat(this.masterTrainingDataset);
		this.masterTrainingDataset = Filter.useFilter(this.masterTrainingDataset,  remove);
		result.setNumMasterDataRecordsAfterDedupe(this.masterTrainingDataset.numInstances());

		AttributeStats stats = this.masterTrainingDataset.attributeStats(this.masterTrainingDataset.numAttributes()-1);
		result.setNumLabelsIdentified(stats.distinctCount);
		
		// Train and save the learner using updated master data set
		this.learner.train(this.masterTrainingDataset);
		
		// Learner is trained and ready to classify again
		
		return result;
	}

	private void buildOracle() throws Exception {
		this.log("_____BUILDING_ORACLE_____");
		System.out.println("Building oracle...");
		// Load the training data for the oracle (should be full set of KDD CUP)
		BufferedReader reader = new BufferedReader(new FileReader(oracleTrainingDatasetFileName));
		Instances oracleTrainingDataset = new Instances(reader);
		
		// Instantiate the oracle
		this.oracle = new Oracle();
		
		// Train the oracle
		String results = this.oracle.train(oracleTrainingDataset);
		// this.log(results);

		// Save as a file on file system so that oracle does not need to be rebuilt with each program execution
		this.oracle.save(oracleFileName);
	}
			
	private Instances createOracleQueryDataset() throws Exception {
		this.log("Entering createOracleQueryDataset...");
		
		Instances queryDataset = new Instances(this.candidateDataset,0);  // Initialize the return set of instances
		
		if (this.candidateDataset != null /* && this.candidateDataset.numInstances() > 0 */) {
			// Return a random small set of instances selected after deduplication

			if(strategy.equals("random")) {
	
				// Remove duplicates
				RemoveDuplicates remove = new RemoveDuplicates();
				remove.setInputFormat(this.candidateDataset);
				this.candidateDataset = Filter.useFilter(this.candidateDataset,  remove);

				if (this.candidateDataset.numInstances() > numInstancesForOracle) {
					// Generate a random number for resampling
					Random rand = new Random(System.nanoTime());
					int seed = rand.nextInt(5000);

					// Create a random sample as a percentage based on number of instances
					double percentSample = (100*numInstancesForOracle)/this.candidateDataset.numInstances();
					Resample resampler = new Resample();
					resampler.setInputFormat(this.candidateDataset);
					resampler.setSampleSizePercent( percentSample );
					resampler.setRandomSeed(seed);
					queryDataset = Filter.useFilter(this.candidateDataset, resampler);
				}
				else {
					// Dataset numInstancesForOracle or less, so just pass it back to the oracle
					queryDataset = new Instances(this.candidateDataset);
				}
			}
			else if(strategy.equals("cluster")) {


				// Remove duplicates
				RemoveDuplicates remove = new RemoveDuplicates();
				remove.setInputFormat(this.candidateDataset);
				this.candidateDataset = Filter.useFilter(this.candidateDataset,  remove);
	
				if (this.candidateDataset.numInstances()>0) {
					Double numInstances = new Double(numInstancesForOracle);
					int numClusters = numInstances.intValue();
					queryDataset = this.getStratifiedInstances(this.candidateDataset, numClusters);
				}
				else {
					queryDataset = new Instances(this.candidateDataset);
				}
				
			}
		}
		else {
			// There were no candidates. Return an empty dataset
			queryDataset = new Instances(this.candidateDataset, 0);
		}
		return queryDataset;		
	}
	
	public Instances getStratifiedInstances (Instances dataset, int numClusters) throws Exception {

		Instances returnDataset = new Instances(dataset, 0);
		
		// Clear the class attribute for k-means
		dataset.setClassIndex(-1);
		
		// Convert double number of instances for oracle to integer
		// Double numInstances = new Double(this.numInstancesForOracle);
		// int numClusters = numInstances.intValue();

		// Get random value for seed
		Random rand = new Random(System.nanoTime());
		int seed = rand.nextInt(5000);
	
		SimpleKMeans kmeans = new SimpleKMeans();
		kmeans.setSeed(seed); 
		kmeans.setPreserveInstancesOrder(true);
		kmeans.setNumClusters(numClusters);
		kmeans.buildClusterer(dataset);
		
		// Get the assignments to clusters
		int[] assignments = kmeans.getAssignments();
		Integer[] sample = new Integer[numClusters];
		for (int i = 0; i < assignments.length; i++) {
			// Linear - may be expensive, but it will ensure that no clusters are missed
			int clusterNum = assignments[i];
			if (sample[clusterNum] == null) {
				sample[clusterNum] = i;	// Assign the first found instance to the sample
				returnDataset.add(this.candidateDataset.instance(i));
			}
		}
		returnDataset.setClassIndex(this.candidateDataset.numAttributes()-1);
		return returnDataset;
	}
	
	private void log(String message) {
		// System.out.println(message);
	}
	
	public void saveDatafile (String fileName, Instances dataset) throws IOException {
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataset);
		saver.setFile(new File(fileName));
		saver.writeBatch();
	}
	
	public void saveLearner (String fileName) throws IOException {
		this.learner.save(fileName);
	}
	
	public Instances getMasterTrainingDataset() {
		return masterTrainingDataset;
	}
	

}


