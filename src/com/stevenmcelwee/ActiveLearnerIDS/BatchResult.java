package com.stevenmcelwee.ActiveLearnerIDS;

import java.util.HashSet;
import java.util.Set;

public class BatchResult {
	private int 	batchNumber;
	private int 	batchSize;			// Adjust for testing
	private String	strategy;			// Strategy for sampling data to send to query
	private double 	minConfidence;		// Adjust for testing
	private double 	maxOracleQuerySize;	// Adjust for testing
	private int 	numLowConfidenceResults;
	private int 	actualOracleQuerySize;
	public void setNumLabelsIdentified(int numLabelsIdentified) {
		this.numLabelsIdentified = numLabelsIdentified;
	}
	private int 	numMasterDataRecordsAfterDedupe;
	private int 	numLabelsIdentified = 0;
	private Set<String> labelsIdentified = new HashSet<String>();

	public BatchResult() {
		super();
	}
	public BatchResult(int batchNumber, int batchSize) {
		super();
		this.batchNumber = batchNumber;
		this.batchSize = batchSize;
	}

	public String getHeader() {
		return "BatchNum,BatchSize,Strategy,MinConfidence,MaxQuerySize,NumLowConf,QuerySize,NumFinalMasterRecs,TotalNumLabels";
	}
	public String toString() {
		String resultString = "" +
				this.batchNumber + "," +
				this.batchSize + "," +
				this.strategy + "," +
				this.minConfidence +  "," +
				this.maxOracleQuerySize +  "," +
				this.numLowConfidenceResults +  "," +
				this.actualOracleQuerySize +  "," +
				this.numMasterDataRecordsAfterDedupe +  "," +
				this.numLabelsIdentified;
		/* for (String label: this.labelsIdentified) {
			resultString += label + ":";			
		} */
		return resultString;
	}
	
	public void addLabelIdentified(String label) {
		this.labelsIdentified.add(label);
		this.numLabelsIdentified++;
	}
	public int getBatchNumber() {
		return batchNumber;
	}
	public void setBatchNumber(int batchNumber) {
		this.batchNumber = batchNumber;
	}
	public int getBatchSize() {
		return batchSize;
	}
	public void setBatchSize(int batchSize) {
		this.batchSize = batchSize;
	}
	public double getMinConfidence() {
		return minConfidence;
	}
	public void setMinConfidence(double minConfidence) {
		this.minConfidence = minConfidence;
	}
	public double getMaxOracleQuerySize() {
		return maxOracleQuerySize;
	}
	public void setMaxOracleQuerySize(double maxOracleQuerySize) {
		this.maxOracleQuerySize = maxOracleQuerySize;
	}
	public int getNumLowConfidenceResults() {
		return numLowConfidenceResults;
	}
	public void setNumLowConfidenceResults(int numLowConfidenceResults) {
		this.numLowConfidenceResults = numLowConfidenceResults;
	}
	public int getActualOracleQuerySize() {
		return actualOracleQuerySize;
	}
	public void setActualOracleQuerySize(int actualOracleQuerySize) {
		this.actualOracleQuerySize = actualOracleQuerySize;
	}
	public int getNumMasterDataRecordsAfterDedupe() {
		return numMasterDataRecordsAfterDedupe;
	}
	public void setNumMasterDataRecordsAfterDedupe(int numMasterDataRecordsAfterDedupe) {
		this.numMasterDataRecordsAfterDedupe = numMasterDataRecordsAfterDedupe;
	}
	public int getNumLabelsIdentified() {
		return numLabelsIdentified;
	}
	public String getStrategy() {
		return strategy;
	}
	public void setStrategy(String strategy) {
		this.strategy = strategy;
	}

}
