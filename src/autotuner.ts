import * as tensorflow from '@tensorflow/tfjs';
import { ModelDict, SequentialModelParameters, DataPoint, BaysianOptimisationStep, LossFunction, ObservedValues, DomainPointValue } from '../types/types';
import * as bayesianOptimizer from './bayesianOptimizer';
import * as gridSearchOptimizer from './gridSearchOptimizer';
import * as paramspace from './paramspace';
import * as priors from './priors';
import * as modelEvaluator from './modelEvaluater';
import { argmax } from './util';

class AutotunerBaseClass {
    metrics: string[] = [];
    observedValues: ObservedValues = {};
    /**
     * Fraction of domain indices that should be evaluated at most
     */
    maxIterations: number;
    /**
     * Predicate that iterates over the observed values and determines wheter or not to stop the tuning of hyperparameters
     * 
     * @return {boolean} false if tuning the hyperparameters should be stopped, true otherwise
     */
    stoppingCriteria: (observedValues: ObservedValues, expectedImprovement: number) => boolean;
    modelOptimizersDict: { [model: string]: tensorflow.Optimizer[] } = {};
    paramspace: any;
    optimizer: any;
    priors: any;
    modelEvaluator: any;

    /**
     * Returns the value of a domain point.
     * 
     * @param {number} domainIndex Index of the domain point to be evaluated.
     * @return {Promise<number>} Value of the domain point
     */
    evaluateModel: (domainIndex: number, objective: string, useCrossValidation: boolean, useTestData: boolean) => Promise<number>;

    bestParameter(objective: string) {
        if (this.checkObjective(objective)) {
            return
        }
        const observedValues = this.observedValues[objective];
        const bestScoreIndex = argmax(observedValues);
        const bestScoreDomainIndex = this.observedValues['domainIndices'][bestScoreIndex];
        return bestScoreDomainIndex;
    }

    /**
     * Tests the best parameters that were found during optimization on the test set.
     * 
     * @param {string} objective Define the metric that should be evaluated. Either 'error' or 'accuracy'
     * @param {boolean} useCrossValidation Indicate wheter or not to use cross validation to evaluate the model. Set to 'false' by default.
     * @return {number} Returns the score of.
     */
    async evaluateBestParameter(objective: string, useCrossValidation: boolean = false) {
        var bestScoreDomainIndex = this.bestParameter(objective) as number;
        return await this.evaluateModel(bestScoreDomainIndex, objective, useCrossValidation, true);
    }

    /**
     * Decide whether to continue tuning the hyperparameters.
     * Stop tuning the parameters if either the maximun muber of iterations has been reached or if 'metricsStopingCriteria' returns false
     * 
     * @return {boolean} true if tuning the hyperparameters should be stopped, false otherwise
     */
    stopTraining(expectedImprovement: number) {
        const domainSize = this.paramspace.domainIndices.length;
        const numberOfObservedValues = this.observedValues['domainIndices'].length;
        var fractionOfEvaluatedPoints = numberOfObservedValues / domainSize;
        var maxIterationsReached: boolean = fractionOfEvaluatedPoints <= this.maxIterations;

        if (this.stoppingCriteria) {
            return maxIterationsReached || this.stoppingCriteria(this.observedValues, expectedImprovement);
        }
        return maxIterationsReached;
    }

    initializePriors() {
        if (!this.priors) {
            this.priors = new priors.Priors(this.paramspace.domainIndices);
        }
    }

    checkObjective (objective: string): boolean {
        const allowedObjectives = ['error', 'accuracy'];
        if (!allowedObjectives.includes(objective)) {
            console.log("Invalid objective function selected!");
            console.log("Objective function must be one of the following: " + allowedObjectives.join());
            return true;
        }
        return false;
    }

    constructor(metrics: string[], dataSet: DataPoint[], numberOfCategories: number, validationSetRatio: number = 0.25, testSetRatio: number = 0.25) {
        this.paramspace = new paramspace.Paramspace();
        this.modelEvaluator = new modelEvaluator.ModelEvaluater(dataSet, numberOfCategories, validationSetRatio, testSetRatio);
        this.metrics = metrics;
        this.observedValues['domainIndices'] = [];
        this.observedValues['error'] = [];
        this.observedValues['accuracy'] = [];
    }

    /**
     * Search the best Parameters using bayesian optimization.
     * 
     * @param {string} [objective='error'] Define the objective of the optimization. Set to 'error' by default.
     * @param {boolean} [useCrossValidation=false] Indicate wheter or not to use cross validation to evaluate the model. Set to 'false' by default.
     * @param {number} [maxIteration=0.75] Fraction of domain points that should be evaluated at most. (e.g. for 'maxIteration=0.75' the optimization stops if 75% of the domain has been evaluated)
     * @param {boolean} [stoppingCriteria] Predicate on the observed values when to stop the optimization
     * @return Returns the best parameters found in the optimization run.
     */
    async bayesianOptimization(objective: string = 'error', useCrossValidation: boolean = false, maxIteration: number = 0.75, stoppingCriteria?: ((observedValues: ObservedValues, expectedImprovement: number) => boolean)) {
        if (this.checkObjective(objective)) {
            return;
        }
        this.initializePriors();
        this.optimizer = new bayesianOptimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains, this.priors.mean, this.priors.kernel);
        this.maxIterations = maxIteration;
        if (stoppingCriteria) {
            this.stoppingCriteria = stoppingCriteria;
        }
        
        return await this.tuneHyperparameters(objective, useCrossValidation);
    }

    /**
     * Search the best Parameters using grid search.
     * 
     * @param {string} [objective='error'] Define the objective of the optimization. Set to 'error' by default.
     * @param {boolean} [useCrossValidation=false] Indicate wheter or not to use cross validation to evaluate the model. Set to 'false' by default.
     * @return Returns the best parameters found in the optimization run.
     */
    async gridSearchOptimizytion(objective: string = 'error', useCrossValidation: boolean = false) {
        if (this.checkObjective(objective)) {
            return;
        }
        this.initializePriors();
        this.optimizer = new gridSearchOptimizer.Optimizer(this.paramspace.domainIndices, this.paramspace.modelsDomains);
        this.maxIterations = 1;

        return await this.tuneHyperparameters(objective, useCrossValidation);
    }


    async tuneHyperparameters(objective: string, useCrossValidation: boolean) {
        console.log("============================");
        console.log("tuning the hyperparameters");

        let optimizing = true;
        while (optimizing) {
            // get the next point to evaluate from the optimizer
            var nextOptimizationPoint: BaysianOptimisationStep = this.optimizer.getNextPoint();
            
            // Train a model given the params and obtain a quality metric value.
            var value = await this.evaluateModel(nextOptimizationPoint.nextPoint, objective, useCrossValidation, false);
            
            // Report the obtained quality metric value.
            this.optimizer.addSample(nextOptimizationPoint.nextPoint, value);

            optimizing = this.stopTraining(nextOptimizationPoint.expectedImprovement);
        }
        // keep observations for the next optimization run
        this.priors.commit(this.paramspace.observedValues);
        
        console.log("============================");
        console.log("finished tuning the hyperparameters");
        console.log();
        var bestScoreDomainIndex = this.bestParameter(objective) as number;
        var bestParameters = this.paramspace.domain[bestScoreDomainIndex]['params'];
        console.log("The best parameters found are:");
        console.log(bestParameters);
        return bestParameters;
    }
}

class TensorflowlModelAutotuner extends AutotunerBaseClass {
    modelDict: ModelDict = {};

    /**
     * Initialize the autotuner.
     * 
     * @param {string[]} metrics
     * @param {DataPoint[]} dataSet
     * @param {number} numberOfCategories 
     * @param {number=0.25} validationSetRatio
     * @param {number=0.25} testSetRatio
     */
    constructor(metrics: string[], dataSet: DataPoint[], numberOfCategories: number, validationSetRatio: number = 0.25, testSetRatio: number = 0.25) {
        super(metrics, dataSet, numberOfCategories, validationSetRatio, testSetRatio);

        this.evaluateModel = async (point: number, objective: string, useCrossValidation: boolean, useTestData: boolean = false) => {
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

            let domainPointValue: DomainPointValue = useCrossValidation 
                ? await this.modelEvaluator.EvaluateSequentialTensorflowModelCV(model, args, useTestData)
                : await this.modelEvaluator.EvaluateSequentialTensorflowModel(model, args, useTestData);

            this.observedValues['domainIndices'].push(point);
            this.observedValues['error'].push(domainPointValue.error);
            this.observedValues['accuracy'].push(domainPointValue.accuracy);
            return objective === 'error' ? domainPointValue.error : 1 - domainPointValue.accuracy;
        } 
    }

    /**
     * Add a new model and its range of parameters to the autotuner.
     * 
     * @param {string} modelIdentifier Identifier of the model
     * @param {tensorflow.Sequential} model Actual Tensorflow model
     * @param {SequentialModelParameters} modelParameters Parameters of the Model: define lossfunction, optimizer, algorithm batch size and number of traning epochs.
     */
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