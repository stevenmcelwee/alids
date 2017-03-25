# alids
Active Learning Intrusion Detection System

This repository contains the code and results of the Active Learning Intrusion Detection 
system developed for the conference paper:

McElwee, S. (2017). Active learning for intrusion detection using k-means clustering 
     selection. In SoutheastCon, 2017, 1-7. IEEE.

Compiling and use:

Compiling:
1. Ensure that the Weka 3.7.13 libraries are installed in your classpath.
2. Compile the source files in src/stevenmcelwee/ActiverLearnerIDS
3. Create a single jar file, such as ActiveLearnerIDS.jar

Executing:
From the command line:

     java -jar ActiveLearnerIDS.jar

Requirements:
Refer to alids.prop file for the list of configurable properties. Most important is
the KDD Cup datafiles that are needed - they must be in ARFF format.
This code was tested with Java 1.8 (64-bit version).
This code was built for and tested with Weka 3.7.13.
Ensure that a large heap size is used. Testing of the prototype used a 4GB heap.

