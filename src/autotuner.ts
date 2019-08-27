import * as tensorflow from '@tensorflow/tfjs';
import * as math from 'mathjs';
import { DataSet, ModelDict, SequentialModelParameters, datasetType, BaysianOptimisationStep, LossFunction, DomainPointValue } from '../types/types';
import * as bayesianOptimizer from './bayesianOptimizer';
import * as gridSearchOptimizer from './gridSearchOptimizer';
import * as paramspace from './paramspace';
import * as priors from './priors';

class AutotunerBaseClass {
    dataset: DataSet;
    metrics: string[] = [];
    observedValues: DomainPointValue[] = [];
    /**
     * Fraction of domain indices that should be evaluated at most
     */
    maxIterations: number;
    /**
     * Predicate that iterates over the observed values and determines wheter or not to stop the tuning of hyperparameters
     * 
     * @return {boolean} false if tuning the hyperparameters should be stopped, true otherwise
     */
    metricsStopingCriteria: (observedValues: DomainPointValue[]) => boolean;
    modelOptimizersDict: { [model: string]: tensorflow.Optimizer[] } = {};
    paramspace: any;
    optimizer: any;
    priors: any;

    /**
     * Returns the value of a domain point.
     * 
     * @param {number} domainIndex Index of the domain point to be evaluated.
     * @return {Promise<number>} Value of the domain point
     */
    evaluateModel: (domainIndex: number) => Promise<number>;

    /**
     * Decide whether to continue tuning the hyperparameters.
     * Stop tuning the parameters if either the maximun muber of iterations has been reached or if 'metricsStopingCriteria' returns false
     * 
     * @return {boolean} false if tuning the hyperparameters should be stopped, true otherwise
     */
    stopingCriteria() {
        const domainSize = this.paramspace.domainIndices.length;
        const numberOfObservedValues = this.observedValues.length;
        var fractionOfEvaluatedPoints = numberOfObservedValues / domainSize;
        var maxIterationsReached: boolean = fractionOfEvaluatedPoints <= this.maxIterations;

        if (this.metricsStopingCriteria) {
            return maxIterationsReached || this.metricsStopingCriteria(this.observedValues);
        }
        return maxIterationsReached;
    }

    initializePriors() {
        if (!this.priors) {
            this.priors = new priors.Priors(this.paramspace.domainIndices);
        }
    }

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType, numberOfCategories: number) {
        this.paramspace = new paramspace.Paramspace();
        this.metrics = metrics;

        const dataset: DataSet = {trainingSet: trainingSet, testSet: testSet, evaluationSet: evaluationSet, numberOfCategories: numberOfCategories};
        this.dataset = dataset;
    }

    /**
     * Search the best Parameters using bayesian optimization.
     * 
     * @param {number} [maxIteration=0.75] Fraction of domain points that should be evaluated at most. (e.g. for 'maxIteration=0.75' the optimization stops if 75% of the domain has been evaluated)
     * @param {boolean} [stopingCriteria] Predicate on the observed values when to stop the optimization
     */
    async bayesianOptimization(maxIteration: number = 0.75, stopingCriteria?: ((observedValues: DomainPointValue[]) => boolean)) {
        this.initializePriors();
        this.optimizer = new bayesianOptimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains, this.priors.mean, this.priors.kernel);
        this.maxIterations = maxIteration;
        if (stopingCriteria) {
            this.metricsStopingCriteria = stopingCriteria;
        }
        
        this.tuneHyperparameters();
    }

    /**
     * Search the best Parameters using grid search.
     */
    async gridSearchOptimizytion() {
        this.initializePriors();
        this.optimizer = new gridSearchOptimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains);
        this.maxIterations = 1;

        this.tuneHyperparameters();
    }


    async tuneHyperparameters() {
        console.log("============================");
        console.log("tuning the hyperparameters");

        let optimizing = true;
        while (optimizing) {
            // get the next point to evaluate from the optimizer
            var nextOptimizationPoint: BaysianOptimisationStep = this.optimizer.getNextPoint();
            
            // Train a model given the params and obtain a quality metric value.
            var value = await this.evaluateModel(nextOptimizationPoint.nextPoint);
            
            // Report the obtained quality metric value.
            this.optimizer.addSample(nextOptimizationPoint.nextPoint, value);

            optimizing = this.stopingCriteria();
            
        }
        // keep observations for the next optimization run
        this.priors.commit(this.paramspace.observedValues);
        
        console.log("============================");
        console.log("finished tuning the hyperparameters");
    }
}

class TensorflowlModelAutotuner extends AutotunerBaseClass {
    modelDict: ModelDict = {};

    constructor(metrics: string[], trainingSet: datasetType, testSet: datasetType, evaluationSet: datasetType, numberOfCategories: number) {
        super(metrics, trainingSet, testSet, evaluationSet, numberOfCategories);

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
          
            let concatenatedTensorTrainData = tensorflow.tidy(() => tensorflow.concat(this.dataset.trainingSet.data));
            let concatenatedTrainLableData = tensorflow.tidy(() => tensorflow.oneHot(this.dataset.trainingSet.lables, this.dataset.numberOfCategories));
            await model.fit(concatenatedTensorTrainData, concatenatedTrainLableData, args);

            let concatenatedTensorTestData = tensorflow.tidy(() => tensorflow.concat(this.dataset.trainingSet.data));
            let concatenatedTestLables = tensorflow.tidy(() => tensorflow.oneHot(this.dataset.trainingSet.lables, this.dataset.numberOfCategories));
            const evaluationResult = model.evaluate(concatenatedTensorTestData, concatenatedTestLables) as tensorflow.Tensor[];

            const error = evaluationResult[0].dataSync()[0];
            const score = evaluationResult[1].dataSync()[0];
            // keep track of the scores
            this.observedValues.push({error: error, metricScores: [score]});
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