import equal from 'deep-equal';
import * as math from 'mathjs';
import { argmax, expectedImprovement } from './util';
import { BaysianOptimisationStep, ModelsDomain, NullableMatrix } from '../types/types';

class Optimizer {
    domainIndices: number[];
    modelsDomains: ModelsDomain;

    modelsSamples: { [identifier: string]: number[]};
    allSamples: number[];
    observedValues: { [identifier: number]: number};
    best: number;
    mean: math.Matrix;
    kernel: math.Matrix;

    constructor (domainIndices: number[], modelsDomains: ModelsDomain, mean: NullableMatrix = null, kernel: NullableMatrix = null) {
        this.domainIndices = domainIndices;
        this.modelsDomains = modelsDomains;

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
        var pointIndex: number = this.domainIndices.findIndex((x: number) => equal(x, point));

        for (var model in this.modelsDomains) {
            if (this.modelsDomains[model].findIndex((x: number) => equal(x, point)) >= 0) {
                this.modelsSamples[model] = this.modelsSamples[model].concat([pointIndex]);
            }
        }

        this.allSamples = this.allSamples.concat([pointIndex]);
        this.observedValues[point] = value;

        if (!this.best || this.observedValues[this.best] < value) {
            this.best = point;
        }
    };

    getNextPoint () {
        var posteriorMean: math.Matrix;
        var posteriorStd: math.Matrix;
        var expectedImprov: math.Matrix;
        var modelExpectedImprovArray: number[] = [];

        // If the whole domain has aleady been sampled, the expected improvement is zero
        let unsampledDataPoints = this.domainIndices.filter(item => this.allSamples.indexOf(item) < 0);
        if (unsampledDataPoints.length === 0) {
            return { nextPoint: 0, expectedImprovement: -2};
        }

        // if no samples have been added yet (e.g. call 'getNextPoint()' the first time) just pick anyone
        if (this.allSamples.length === 0){
            return { nextPoint: this.domainIndices[0], expectedImprovement: -1}
        }

        // Compute best rewards for each model.
        var modelsBestRewards: { [model: string]: number } = {};
        for (let model in this.modelsSamples) {
            modelsBestRewards[model] = Math.max.apply(null, Array.from(this.modelsSamples[model], (x: number) => this.observedValues[this.domainIndices[x]]));
        }

        // Compute posterior distribution (mean and standard deviation).
        var domainSize: number = this.mean.size()[0];
        var sampleSize: number = this.allSamples.length;
        var sampleRewards: math.Matrix = math.matrix(Array.from(this.allSamples, (x: number) => this.observedValues[this.domainIndices[x]]));
        var samplePriorMean: math.Matrix = this.mean.subset(math.index(this.allSamples));

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
        var sampleRewardGain = math.reshape(math.subtract(sampleRewards, samplePriorMean) as math.Matrix, [sampleSize, 1]);
        var sampleKernelDotGain = math.multiply(sampleKernelInv, sampleRewardGain) as math.Matrix;
        
        posteriorMean = math.add(math.multiply(allToSampleKernel, sampleKernelDotGain) as math.Matrix, math.reshape(this.mean, [domainSize, 1]) as math.Matrix) as math.Matrix;
        // reshape the mean back to its original size
        math.reshape(this.mean, [domainSize])

        var posteriorKernel = math.multiply(allToSampleKernel, math.multiply(sampleKernelInv, math.transpose(allToSampleKernel) as math.Matrix)) as math.Matrix;
        posteriorKernel = math.subtract(this.kernel, posteriorKernel) as math.Matrix;

        posteriorStd = math.sqrt(math.reshape(math.diag(posteriorKernel), ([domainSize, 1])) as math.Matrix);

        // Compute the expected improvement.
        expectedImprov = math.zeros(domainSize) as math.Matrix;
        for (let model in this.modelsDomains) {
            // exclude obtained samples
            var modelPoints = this.modelsDomains[model].filter(item => this.modelsSamples[model].indexOf(item) < 0);
            var modelPosteriorMean = posteriorMean.subset(math.index(modelPoints, 0));
            var modelPosteriorStd = posteriorStd.subset(math.index(modelPoints, 0));
            var modelExpectedImprov = expectedImprovement(modelsBestRewards[model], modelPosteriorMean, modelPosteriorStd) as math.Matrix;

            if (typeof(modelExpectedImprov) === 'number') {
                // TODO: fix this corner case according to the regular case
                // retrieve 'var improvement = math.add(expectedImprov.subset(math.index(modelPoints)), modelExpectedImprov);' as a number to avoid TypeError
                expectedImprov = expectedImprov.subset(math.index(modelPoints), modelExpectedImprov) as math.Matrix;
            } else {
                modelExpectedImprov = math.reshape(modelExpectedImprov, [modelPoints.length]) as math.Matrix;
                var improvement = math.add(expectedImprov.subset(math.index(modelPoints)), modelExpectedImprov);
                expectedImprov = expectedImprov.subset(math.index(modelPoints), improvement) as math.Matrix;
            }
        }

        for (var i = 0; i < this.allSamples.length; i++) {
            modelExpectedImprovArray[this.allSamples[i]] = 0;
        }

        var domain: number[] = [];
        for (let k in this.modelsDomains) {
            domain = domain.concat(this.modelsDomains[k])
        }

        // Sample the point with maximal expected improvement over the given domain.
        var idx = argmax(math.subset(expectedImprov.toArray(), math.index(domain)) as number[]);

        // return the point with the biggest expected improvement as well as the expexted improvement
        const baysianOptimisationStep: BaysianOptimisationStep = { nextPoint: this.domainIndices[domain[idx]], expectedImprovement: modelExpectedImprovArray[idx]}
        return baysianOptimisationStep;
    };
}

export { Optimizer };