/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

package machinelearninglabs;

import weka.classifiers.Classifier;
import weka.core.Capabilities;
import weka.core.Instance;
import weka.core.Instances;

/**
 *
 * @author aku12vtu
 */
public class OENaiveBayesClassifier implements Classifier {

    private int[] classCount;
    private double[] classProbs;
    
    private int[][][] allAttributeCounts;
    private double[][][] conditionalProbabilities;
    
    @Override
    public void buildClassifier(Instances data) throws Exception {
        allAttributeCounts = new int[data.numAttributes()-1][][]; // I minused 1 because of the class attr
        conditionalProbabilities = new double[data.numAttributes()-1][][];
        
        for(int i = 0; i < data.numAttributes()-1; i++){
            allAttributeCounts[i] = attributeCounts(data, i);
        }
        
        for(int i = 0; i < data.numAttributes()-1; i++){
            conditionalProbabilities[i] = attributeProbs(data, i); 
            printDoubleMatrix(conditionalProbabilities[i]);
        }
        
        
        
    }

    @Override
    public double classifyInstance(Instance instance) throws Exception {
        double[] probabilities = distributionForInstance(instance);
        
        double maxSoFar = 0;
        int maxIndex = 0;
        for(int i = 0; i < probabilities.length; i++){
            if(probabilities[i] > maxSoFar){
                maxSoFar = probabilities[i];
                maxIndex = i;
            }
        }
        
        return maxIndex;
    }

    @Override
    public double[] distributionForInstance(Instance instance) throws Exception {
        // return an array with a size of the number of classes
        double[] jointProbabilities = new double[instance.attribute(instance.classIndex()).numValues()];
        double[] result = new double[instance.attribute(instance.classIndex()).numValues()];
        
        // calculate un-normaalized probs
        for(int cls = 0; cls < jointProbabilities.length; cls++){
            double p = classProbs[cls];
            for (int att = 0; att < instance.numAttributes()-1; att++)
            {
                int value = (int) instance.value(att);
                p *= conditionalProbabilities[att][cls][value];
            }
            jointProbabilities[cls] = p;
        }

        // Find normalized probabilities
        for(int i = 0; i < jointProbabilities.length; i++){
            double denominator = 0;
            for(int j = 0; j < jointProbabilities.length; j++){
                denominator += jointProbabilities[j];
            }
            result[i] = jointProbabilities[i]/denominator;
        }
        return result;
    }
    
    @Override
    public Capabilities getCapabilities() {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    
    /***************************** HELPER METHODS  ************************************/
    
    public void classDistribution(Instances data){
        
        classCount = new int[data.firstInstance().numClasses()];
        classProbs = new double[data.firstInstance().numClasses()];

        // Get the frequency/count of each class in the data
        for(Instance eachInstance : data){
                double classValue = eachInstance.value(eachInstance.classIndex());
                classCount[(int)classValue]++;
        }
        
        // Get the probability of the occurence of each class
        for(int i = 0; i < classProbs.length; i++){
            classProbs[i] = (double)classCount[i]/data.numInstances();
        }
        
        printIntArray(classCount);
        System.out.println(data.firstInstance().value(0));
        printDoubleArray(classProbs);
    }
    
    public int[][] attributeCounts(Instances data, int att) {
        int numberOfPossibleValuesForAttribute = data.firstInstance().attribute(att).numValues();
        int[][] result = new int[data.numClasses()][numberOfPossibleValuesForAttribute];
        
        // for each class
        for(Instance eachInstance : data){
                double classValue = eachInstance.value(eachInstance.classIndex());
                result[(int)classValue][(int)eachInstance.value(att)]++;
        }
        //printIntMatrix(result);
        return result;
    }
    
    public double[][] attributeProbs(Instances data, int att) {
        int numberOfPossibleValuesForAttribute = data.firstInstance().attribute(att).numValues();
        double[][] result = new double[data.numClasses()][numberOfPossibleValuesForAttribute];
        
        // for each class
        for(Instance eachInstance : data){
                double classValue = eachInstance.value(eachInstance.classIndex());
                result[(int)classValue][(int)eachInstance.value(att)]++;
        }
        
        // Get conditional probabilities ie probability that attribute = x given some class
        for(int i = 0; i < result.length; i++){
            for(int j = 0; j < result[i].length; j++){
                result[i][j] = (double) result[i][j] / classCount[i];
            }
        }
        //printDoubleMatrix(result);
        return result;
    }
    
    
    /***************************** PRINT METHODS  ************************************/
    
    public void printIntArray(int[] array){
        for(int i = 0; i < array.length; i++){
            System.out.println(array[i]);
        }
    }
    
    public void printDoubleArray(double[] array){
        for(int i = 0; i < array.length; i++){
            System.out.println(array[i]);
        }
    }
    
    public void printIntMatrix(int[][] array){
        for(int i = 0; i < array.length; i++){
            for(int j = 0; j < array[i].length; j++){
                System.out.println(array[i][j]);
            }
            System.out.println("");
        }
    }
    
    public void printDoubleMatrix(double[][] array){
        for(int i = 0; i < array.length; i++){
            for(int j = 0; j < array[i].length; j++){
                System.out.println(array[i][j]);
            }
            System.out.println("");
        }
    }

    
    
}
