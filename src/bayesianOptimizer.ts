import * as math from 'mathjs';
import { argmax, expectedImprovement } from './util';
import { BaysianOptimisationStep, ModelsDomain, NullableMatrix } from '../types/types';

class Optimizer {
    domainIndices: number[];
    modelsDomains: ModelsDomain;
    acquisitionFunction: string;

    modelsSamples: { [identifier: string]: number[]};
    allSamples: number[];
    observedValues: { [identifier: number]: number};
    mean: math.Matrix;
    kernel: math.Matrix;

    constructor (domainIndices: number[], modelsDomains: ModelsDomain, acquisitionFunction: string, mean: NullableMatrix = null, kernel: NullableMatrix = null) {
        this.domainIndices = domainIndices;
        this.modelsDomains = modelsDomains;
        this.acquisitionFunction = acquisitionFunction;

        // contains dictionary that maps from model to sampled data points
        this.modelsSamples = {};
        for (var model in modelsDomains) {
            this.modelsSamples[model] = [];
        }
        this.allSamples = [];
        this.observedValues = {};

        const domainSize: number = Object.keys(this.domainIndices).length;

        if (mean === null) {
            this.mean = math.zeros(domainSize) as math.Matrix;
        } else {
            this.mean = mean;
        }
        
        if (kernel === null) {
            this.kernel = math.identity(domainSize) as math.Matrix;
        } else {
            this.kernel = kernel;
        }
    }

    addSample (point: number, value: number) {
        var pointIndex: number = this.domainIndices.findIndex((x: number) => (x === point));

        for (var model in this.modelsDomains) {
            if (this.modelsDomains[model].findIndex((x: number) => (x === point)) >= 0) {
                this.modelsSamples[model] = this.modelsSamples[model].concat([pointIndex]);
            }
        }

        this.allSamples = this.allSamples.concat([pointIndex]);
        this.observedValues[point] = value;
    };

    getNextPoint () {
        var posteriorMean: math.Matrix;
        var posteriorStd: math.Matrix;
        var expectedImprov: math.Matrix;
        var upperConfidenceBounds: math.Matrix;
        var modelExpectedImprovArray: number[] = [];
        var upperConfidenceBoundsArray: number[] = [];

        // if no samples have been added yet (e.g. call 'getNextPoint()' the first time) just pick anyone
        if (this.allSamples.length === 0){
            return { nextPoint: this.domainIndices[0], acquisitionFunctionValue: -1}
        }

        // Compute best rewards for each model.
        var modelsBestRewards: { [model: string]: number } = {};
        for (let model in this.modelsSamples) {
            modelsBestRewards[model] = Math.max.apply(null, Array.from(this.modelsSamples[model], (x: number) => this.observedValues[this.domainIndices[x]]));
        }

        // Compute posterior distribution (mean and standard deviation).
        var domainSize: number = this.mean.size()[0];
        var sampleSize: number = this.allSamples.length;
        var meanArray = this.mean.toArray() as number[];

        var sampleKernel: math.Matrix = this.kernel.subset(math.index(this.allSamples, this.allSamples));
        var allToSampleKernel: math.Matrix = this.kernel.subset(math.index(math.range(0, domainSize), this.allSamples));

        // TODO: check this case with regards do math.matrix()
        // Sample kernel is sometimes a scalar.
        if (typeof(sampleKernel) === 'number') {
            sampleKernel = math.matrix([[sampleKernel]]);
        }

        // Defend against singular matrix inversion.
        sampleKernel = math.add(sampleKernel, math.multiply(math.identity(sampleSize), 0.001)) as math.Matrix;

        var sampleKernelInv = math.inv(sampleKernel);
        var sampleRewardGain = math.reshape(math.matrix(Array.from(this.allSamples, (x: number) => this.observedValues[this.domainIndices[x]] - meanArray[this.domainIndices[x]])), [sampleSize, 1]);
        var sampleKernelDotGain = math.multiply(sampleKernelInv, sampleRewardGain) as math.Matrix;
        
        posteriorMean = math.multiply(allToSampleKernel, sampleKernelDotGain) as math.Matrix;

        var posteriorKernel = math.multiply(allToSampleKernel, math.multiply(sampleKernelInv, math.transpose(allToSampleKernel) as math.Matrix)) as math.Matrix;
        posteriorKernel = math.subtract(this.kernel, posteriorKernel) as math.Matrix;

        posteriorStd = math.sqrt(math.reshape(math.diag(posteriorKernel), ([domainSize, 1])) as math.Matrix);

        // Compute the expected improvement.
        expectedImprov = math.zeros(domainSize) as math.Matrix;
        upperConfidenceBounds = math.zeros(domainSize) as math.Matrix;
        for (let model in this.modelsDomains) {
            // exclude obtained samples
            var modelPoints = this.modelsDomains[model].filter(item => this.modelsSamples[model].indexOf(item) < 0);
            var modelPosteriorMean = posteriorMean.subset(math.index(modelPoints, 0));
            var modelPosteriorStd = posteriorStd.subset(math.index(modelPoints, 0));

            var modelExpectedImprov = expectedImprovement(modelsBestRewards[model], modelPosteriorMean, modelPosteriorStd) as math.Matrix;
            var modelUpperConfidenceBounds = math.add(modelPosteriorMean, modelPosteriorStd) as math.Matrix;

            if (typeof(modelExpectedImprov) === 'number') {
                // TODO: fix this corner case according to the regular case
                // retrieve 'var improvement = math.add(expectedImprov.subset(math.index(modelPoints)), modelExpectedImprov);' as a number to avoid TypeError
                expectedImprov = expectedImprov.subset(math.index(modelPoints), modelExpectedImprov) as math.Matrix;
                upperConfidenceBounds = upperConfidenceBounds.subset(math.index(modelPoints), modelUpperConfidenceBounds) as math.Matrix;
            } else {
                modelExpectedImprov = math.reshape(modelExpectedImprov, [modelPoints.length]) as math.Matrix;
                var improvement = math.add(expectedImprov.subset(math.index(modelPoints)), modelExpectedImprov);
                expectedImprov = expectedImprov.subset(math.index(modelPoints), improvement) as math.Matrix;

                upperConfidenceBounds = math.reshape(modelUpperConfidenceBounds, [modelPoints.length]) as math.Matrix;
                var upperBound = math.add(upperConfidenceBounds.subset(math.index(modelPoints)), modelUpperConfidenceBounds);
                upperConfidenceBounds = upperConfidenceBounds.subset(math.index(modelPoints), upperBound) as math.Matrix;
            }
        }

        modelExpectedImprovArray = expectedImprov.toArray() as number[];
        for (var i = 0; i < this.allSamples.length; i++) {
            modelExpectedImprovArray[this.allSamples[i]] = 0;
        }

        upperConfidenceBoundsArray = upperConfidenceBounds.toArray() as number[];

        var domain: number[] = [];
        for (let k in this.modelsDomains) {
            domain = domain.concat(this.modelsDomains[k])
        }

        // Compute the index of the next domain point based on the acquisition function
        var idx: number;
        var baysianOptimisationStep: BaysianOptimisationStep;
        if (this.acquisitionFunction === 'expectedImprovement') {
            // Sample the point with maximal expected improvement over the given domain.
            idx = argmax(math.subset(modelExpectedImprovArray, math.index(domain)) as number[]);
            
            // return the point with the biggest expected improvement as well as the expexted improvement
            baysianOptimisationStep = { nextPoint: this.domainIndices[domain[idx]], acquisitionFunctionValue: modelExpectedImprovArray[idx]}
        } else {
            // Sample the point with maximal upper confidence bound.
            idx = argmax(math.subset(upperConfidenceBoundsArray, math.index(domain)) as number[]);

            // return the point with the biggest expected improvement as well as the expexted improvement
            baysianOptimisationStep = { nextPoint: this.domainIndices[domain[idx]], acquisitionFunctionValue: upperConfidenceBoundsArray[idx]}
        }

        return baysianOptimisationStep;
    };
}

export { Optimizer };