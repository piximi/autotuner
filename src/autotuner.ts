import * as tensorflow from '@tensorflow/tfjs';
import { DataSet, ModelDict, SequentialModelParameters, datasetType, BaysianOptimisationStep, LossFunction } from '../types/types';
import  * as optimizer from './optimizer';
import  * as paramspace from './paramspace';
import  * as priors from './priors';

class AutotunerBaseClass {
    dataset: DataSet;
    metrics: string[] = [];

    modelOptimizersDict: { [model: string]: tensorflow.Optimizer[] } = {};

    paramspace: any;
    optimizer: any;
    priors: any;

    evaluateModel: (domainIndex: number) => Promise<number>;

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType) {
        this.paramspace = new paramspace.Paramspace();
        this.metrics = metrics;

        const dataset: DataSet = {trainingSet: trainingSet, testSet: testSet, evaluationSet: evaluationSet};
        this.dataset = dataset;
    }

    async tuneHyperparameters(usePriorObservations: boolean = false) {
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
            var value = await this.evaluateModel(nextOptimizationPoint.nextPoint);
            
            // Report the obtained quality metric value.
            this.optimizer.addSample(nextOptimizationPoint.nextPoint, value);

            // keep observations for the next optimization run
            this.priors.commit(this.paramspace.observedValues);
        }
    }
}

class TensorflowlModelAutotuner extends AutotunerBaseClass {
    modelDict: ModelDict = {};

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType) {
        super(metrics, trainingSet, testSet, evaluationSet);

        this.evaluateModel = async (point: number) => {
            const modelIdentifier = this.paramspace.domain[point]['model'];
            const model = this.modelDict[modelIdentifier];
            const params = this.paramspace.domain[point]['params'];

            const args = {
                batchSize: params["batchSize"],
                epochs: params["epochs"]
            };


            const optimizerFunction = this.modelOptimizersDict[modelIdentifier][params["optimizerFunction"]];
            model.compile({
                loss: LossFunction[params["lossFunction"]],
                metrics: metrics,
                optimizer: optimizerFunction
            });
          
            await model.fit(this.dataset.trainingSet.data, this.dataset.trainingSet.lables, args);

            const evaluationResult = model.evaluate(this.dataset.testSet.data, this.dataset.testSet.lables) as tensorflow.Tensor[];
            const score = evaluationResult[1].dataSync()[0];

            return score;
        }
    }

    addModel(modelIdentifier: string, model: tensorflow.Sequential, modelParameters: SequentialModelParameters) {
        this.modelDict[modelIdentifier] = model;

        this.modelOptimizersDict[modelIdentifier] = modelParameters.optimizerAlgorithm;
        const optimizerAlgorithParameters = Array.from(modelParameters.optimizerAlgorithm, (x,i) => i);

        this.paramspace.addModel(modelIdentifier, {
            'lossFunction' : modelParameters.lossfunction, 
            'optimizerFunction' : optimizerAlgorithParameters, 
            'batchSize' : modelParameters.batchSize, 
            'epochs' : modelParameters.epochs});
    }
}

export { AutotunerBaseClass, TensorflowlModelAutotuner }