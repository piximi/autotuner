import * as tensorflow from '@tensorflow/tfjs';
import { CreateModelFunction, DataSet, ModelDict, SequentialModelParameters, datasetType, BaysianOptimisationStep } from '../types/types';
import  * as optimizer from './optimizer';
import  * as paramspace from './paramspace';
import  * as priors from './priors';

class AutotunerBaseClass {
    dataset: any;
    metrics: string[] = [];

    modelOptimizersDict: { [model: string]: tensorflow.Optimizer[] } = {};

    paramspace: any;
    optimizer: any;
    priors: any;

    evaluateModel: (domainIndex: number) => number;

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType) {
        this.paramspace = new paramspace.Paramspace();
        this.metrics = metrics;

        const dataset: DataSet = {trainingSet: trainingSet, testSet: testSet, evaluationSet: evaluationSet};
        this.dataset = dataset;
    }

    tuneHyperparameters(usePriorObservations: boolean = false) {
        if (!usePriorObservations || !this.priors) {
            this.priors = new priors.Priors(this.paramspace.domainIndices);
        }
        this.optimizer = new optimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains, this.priors.mean, this.priors.kernel);

        let optimizing = true;
        while (optimizing) {
            // get the next point to evaluate from the optimizer
            var nextOptimizationPoint: BaysianOptimisationStep = this.optimizer.getNextPoint();

            // check if 'expectedImprovement' === -2, if so there are no more points to evaluate
            if (nextOptimizationPoint.expectedImprovement === -2) {
                break;
            }
            
            // Train a model given the params and obtain a quality metric value.
            var value = this.evaluateModel(nextOptimizationPoint.nextPoint);
            
            // Report the obtained quality metric value.
            this.optimizer.addSample(nextOptimizationPoint.nextPoint, value);

            // keep observations for the next optimization run
            this.priors.commit(this.paramspace.observedValues);
        }
    }
}

class TensorflowlModelAutotuner extends AutotunerBaseClass {
    modelDict: ModelDict = {};

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType){
        super(metrics, trainingSet, testSet, evaluationSet);
        this.evaluateModel = (point: number) => { 
            // TODO: implement
            return 2};
    }

    addModel(modelIdentifier: string, model: tensorflow.Sequential, modelParameters: SequentialModelParameters) {
        this.modelDict[modelIdentifier] = model;

        this.modelOptimizersDict[modelIdentifier] = modelParameters.optimizerAlgorith;
        const optimizerAlgorithParameters = Array.from(modelParameters.optimizerAlgorith, (x,i) => i);

        this.paramspace.addModel(modelIdentifier, {
            'lossFunction' : modelParameters.lossfunction, 
            'optimizerFunction' : optimizerAlgorithParameters, 
            'batchSize' : modelParameters.batchSize, 
            'epochs' : modelParameters.epochs});
    }
}

export { AutotunerBaseClass, TensorflowlModelAutotuner }