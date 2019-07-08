import equal from 'deep-equal';
import difference from 'lodash/difference';
import * as math from 'mathjs';
import { argmax, expectedImprovement } from './util';
import { BaysianOptimisationStep, ModelsDomain, NullableMatrix } from '../types/types';

class Optimizer {
    domainIndices: number[];
    modelsDomains: ModelsDomain;
    modelsSamples: any;
    allSamples: any;
    allSamplesDelays: any;
    observedValues: any;
    best: any;
    delays: any;
    mean: math.Matrix;
    kernel: math.Matrix;
    strategy: any;

    constructor (domainIndices: number[], modelsDomains: ModelsDomain, mean: NullableMatrix = null, kernel: NullableMatrix = null, delays: NullableMatrix = null, strategy='ei') {
    this.domainIndices = domainIndices;
    this.modelsDomains = modelsDomains;

    // contains dictionary that maps from model to sampled data points
    this.modelsSamples = {};
    for (var model in modelsDomains) {
        this.modelsSamples[model] = math.matrix([]);
    }
    this.allSamples = math.matrix([]);
    this.allSamplesDelays = math.matrix([]);
    this.observedValues = {};
    this.best = null;

    const domainSize: number = Object.keys(this.domainIndices).length;

    if (delays === null) {
        this.delays = math.ones(domainSize);
    } else {
        this.delays = delays;
    }
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

    this.strategy = strategy;
    }


    addSample (point: number, value: number, delay: number = 1.0) {
        var pointIndex: number = this.domainIndices.findIndex((x: number) => equal(x, point));

        for (var model in this.modelsDomains) {
            if (this.modelsDomains[model].findIndex((x: number) => equal(x, point)) >= 0) {
                this.modelsSamples[model] = math.concat(this.modelsSamples[model], [pointIndex]);
            }
        }

        this.allSamples = math.concat(this.allSamples, [pointIndex]);
        this.allSamplesDelays = math.concat(this.allSamplesDelays, [delay]);
        this.observedValues[point] = value;

        if (this.best === null || this.observedValues[this.best] < value) {
            this.best = point;
        }
    };


    getNextPoint (excludeModels: string[] = []) {
        var domainIndices = Array.from(new Array(Object.keys(this.domainIndices).length), (x,i) => i);

        var posteriorMean: math.Matrix;
        var posteriorStd: math.Matrix;
        var expectedImprov: math.Matrix;
        var modelExpectedImprovArray: number[] = [];

        // If allSamples contains samples from the whole domain, then we will skip the posterior calculation step.
        if (difference(domainIndices, this.allSamples).length === 0) {
            posteriorMean = math.matrix(Array.from(this.domainIndices, (x) => this.observedValues[x])) as math.Matrix;
            posteriorStd = math.zeros(posteriorMean.size()) as math.Matrix;
            expectedImprov = math.zeros(posteriorMean.size()) as math.Matrix;

        } else {

            // Compute best rewards for each model.
            var modelsBestRewards: { [model: string]: number } = {};
            for (let model in this.modelsSamples) {
                modelsBestRewards[model] = Math.max.apply(null, Array.from(this.modelsSamples[model].toArray(), (x: number) => this.observedValues[this.domainIndices[x]]));
            }

            // Compute posterior distribution (mean and standard deviation).
            var domainSize: number = this.mean.size()[0];
            var sampleSize: number = this.allSamples.size()[0];
            var sampleRewards: math.Matrix = math.matrix(Array.from(this.allSamples.toArray(), (x: number) => this.observedValues[this.domainIndices[x]]));
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

            var posteriorKernel = math.multiply(allToSampleKernel, math.multiply(sampleKernelInv, math.transpose(allToSampleKernel) as math.Matrix)) as math.Matrix;
            posteriorKernel = math.subtract(this.kernel, posteriorKernel) as math.Matrix;

            posteriorStd = math.sqrt(math.reshape(math.diag(posteriorKernel), ([domainSize, 1])) as math.Matrix);

            // Compute the expected improvement.
            expectedImprov = math.zeros(domainSize) as math.Matrix;
            for (let model in this.modelsDomains) {
                // exclude obtained samples
                var modelPoints = difference(this.modelsDomains[model], this.modelsSamples[model].toArray());
                var modelPosteriorMean = posteriorMean.subset(math.index(modelPoints, 0));
                var modelPosteriorStd = posteriorStd.subset(math.index(modelPoints, 0));
                var modelExpectedImprov = expectedImprovement(modelsBestRewards[model], modelPosteriorMean, modelPosteriorStd) as math.Matrix;
                modelExpectedImprov = math.reshape(modelExpectedImprov, [modelPoints.length]) as math.Matrix;

                expectedImprov = expectedImprov.subset(math.index(modelPoints), math.add(expectedImprov.subset(math.index(modelPoints)), modelExpectedImprov)) as math.Matrix;
            }

            // Rescale EI with delays.
            modelExpectedImprovArray = math.dotDivide(expectedImprov, this.delays) as number[];

            for (var i = 0; i < this.allSamples.length; i++) {
                modelExpectedImprovArray[this.allSamples[i]] = 0;
            }
        }

        // Determine the model choice based on the specified strategy.
        var allModels = Object.keys(this.modelsDomains);
        var domain: number[]

        if (this.strategy === 'ei') {
            // Exclude some models from the domain if specified.
            allModels = difference(allModels, excludeModels);
            var excludedDomain: number[] = [];
            for (var i = 0; i < excludeModels.length; i++) {
                excludedDomain = excludedDomain.concat(this.modelsDomains[excludeModels[i]]);
            }
            domain = [];
            for (let k in allModels) {
                domain.concat(this.modelsDomains[k])
            }
        } else if (this.strategy === 'rr') {
            var model = allModels[this.allSamples.length % allModels.length];
            domain = this.modelsDomains[model];
        } else if (this.strategy === 'rnd') {
            var model = allModels[Math.floor(Math.random() * allModels.length)];
            domain = this.modelsDomains[model];
        } else {
            throw "Accepted values for strategy are: ei, rr and rnd.";
        }

        // Sample the point with maximal expected improvement over the given domain.
        var idx = argmax(math.subset(expectedImprov, math.index(domain)) as number[]);

        if (modelExpectedImprovArray[idx] === 0) {
            // If the whole domain has been sampled, then just run the point with the shortest delay.
            idx = argmax(this.delays.subset(math.index(domain)).toArray());
        }

        // return the point with the biggest expected improvement as well as the expexted improvement
        const baysianOptimisationStep: BaysianOptimisationStep = { nextPoint: this.domainIndices[domain[idx]], expectedImprovement: modelExpectedImprovArray[idx]}
        return baysianOptimisationStep;
    };
}

export { Optimizer };