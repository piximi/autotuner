import * as tensorflow from '@tensorflow/tfjs';
import { CreateModelFunction, LossFunction, DataSet, ModelDict, SequentialModelParameters } from '../types/types';
import  * as optimizer from './optimizer';
import  * as paramspace from './paramspace';
import  * as priors from './priors';

class AutotunerBaseClass {
    dataset: DataSet;
    metrics: string[] = [];

    modelLossFunctionsDict: { [model: string]: LossFunction[] } = {};
    modelOptimizersDict: { [model: string]: tensorflow.Optimizer[] } = {};

    paramspace: any;
    opt: any;
    priors: any;

    evaluateModel: (point: number) => number;

    constructor(metrics: string[], trainingSet: tensorflow.Tensor<tensorflow.Rank>, testSet: tensorflow.Tensor<tensorflow.Rank>, evaluationSet: tensorflow.Tensor<tensorflow.Rank>) {
        this.paramspace = new paramspace.Paramspace();
        this.metrics = metrics;

        this.dataset.trainingSet = trainingSet;
        this.dataset.testSet = testSet;
        this.dataset.evaluationSet = evaluationSet;
    }

    tuneHyperparameters(usePriorObservations: boolean = false) {
        if (!usePriorObservations || !this.priors) {
            this.priors = new priors.Priors(this.paramspace.domainIndices);
        }
        this.opt = new optimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains, this.priors.mean, this.priors.kernel);

        let optimizing = false;
        while (optimizing) {
            // Take a suggestion from the optimizer.
            var point = this.opt.getNextPoint();
            
            // Train a model given the params and obtain a quality metric value.
            var value = this.evaluateModel(point);
            
            // Report the obtained quality metric value.
            this.paramspace.addSample(point, value);

            // keep observations for the next optimization run
            this.priors.commit(this.paramspace.observedValues);
        }  
    }
}

class TensorflowlModelAutotuner extends AutotunerBaseClass {
    modelDict: ModelDict = {};

    constructor(metrics: string[], trainingSet: tensorflow.Tensor<tensorflow.Rank>, testSet: tensorflow.Tensor<tensorflow.Rank>, evaluationSet: tensorflow.Tensor<tensorflow.Rank>){
        super(metrics, trainingSet, testSet, evaluationSet);
        this.evaluateModel = (point: number) => { 
            // TODO: implement
            return point};
    }

    addModel(modelIdentifier: string, model: tensorflow.Sequential, modelParameters: SequentialModelParameters) {
        this.modelDict[modelIdentifier] = model;

        this.modelLossFunctionsDict[modelIdentifier] = modelParameters.lossfunction;
        this.modelOptimizersDict[modelIdentifier] = modelParameters.optimizerAlgorith;

        const lossParameters = Array.from(modelParameters.lossfunction, (x,i) => i);
        const optimizerAlgorithParameters = Array.from(modelParameters.optimizerAlgorith, (x,i) => i);

        this.paramspace.addModel(modelIdentifier, {
            'lossFunction' : lossParameters, 
            'optimizerFunction' : optimizerAlgorithParameters, 
            'batchSize' : modelParameters.batchSize, 
            'epochs' : modelParameters.epochs});
    }
}

export { TensorflowlModelAutotuner }