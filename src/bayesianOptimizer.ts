import * as math from 'mathjs';
import { argmax, expectedImprovement, upperConfidenceBound } from './util';
import { BaysianOptimisationStep } from '../types/types';

class Optimizer {
    domainIndices: number[];
    acquisitionFunction: (means: math.Matrix, stds: math.Matrix) => number[];
    best: number;
    
    xObserved: number[] = [];
    xUnobserved: number[] = [];
    yObserved: number[] = [];

    constructor (domainIndices: number[], acquisitionFunction: string) {
        this.domainIndices = domainIndices;
        this.xUnobserved = domainIndices;
        if (acquisitionFunction == 'expectedImprovement') {
            this.acquisitionFunction = (means: math.Matrix, stds: math.Matrix) => expectedImprovement(this.best, means, stds);
        } else {
            this.acquisitionFunction = upperConfidenceBound;
        }
    }

    addSample (point: number, value: number) {
        if (value < this.best || this.best === undefined) {
            this.best = value;
        }

        this.xObserved.push(point);
        this.yObserved.push(value);
        this.xUnobserved = this.xUnobserved.filter( (i: number) => { return i !== point});
    };

    getNextPoint () {
        var idx: number;
        var acquisitionFunctionValue: number;

        const sampleSize = this.xObserved.length;
        const domainSize = this.domainIndices.length;

        // sample 15% of the domain before calculating the GP
        if (sampleSize / domainSize < 0.15) {
            idx = Math.floor(Math.random() * this.xUnobserved.length);
            acquisitionFunctionValue = 0;
        } else {
            const GP = this.evaluateGP();
            const acquisitionFunctionValues = this.acquisitionFunction(GP.means, GP.stds);

            idx = argmax(acquisitionFunctionValues);
            acquisitionFunctionValue = acquisitionFunctionValues[idx]
        }

        const nextIndex = this.xUnobserved[idx];
        return { nextPoint: nextIndex, acquisitionFunctionValue: acquisitionFunctionValue} as BaysianOptimisationStep;
    };

    /**
     * Compute the gaussian process and return the mean and std of all unobserved domain Indices.
     */
    evaluateGP () {
        const sampleSize = this.xObserved.length;

        // compute the kernel matrix components
        const K = math.matrix(this.kernel(this.xObserved, this.xObserved));
        const K_ = math.matrix(this.kernel(this.xObserved, this.xUnobserved));
        const C = math.matrix(this.kernel(this.xUnobserved, this.xUnobserved));

        // Defend against singular matrix inversion.
        var covarMatrix = math.add(K, math.multiply(math.identity(sampleSize), 0.001)) as math.Matrix;
        var invCovarMatrix = math.inv(covarMatrix);
        var sampleGain = math.multiply(invCovarMatrix, math.matrix(this.yObserved)) as math.Matrix;
        // compute mean for all unobserved domain points
        var posteriorMean = math.multiply(math.transpose(K_), math.reshape(sampleGain, [sampleSize, 1])) as math.Matrix;
        
        var posteriorKernel = math.multiply(math.transpose(K_), math.multiply(invCovarMatrix, K_ as math.Matrix)) as math.Matrix;
        posteriorKernel = math.subtract(C, posteriorKernel) as math.Matrix;
        // compute std for all unobserved domain points
        var posteriorStd = math.sqrt(math.reshape(math.diag(posteriorKernel), ([this.xUnobserved.length, 1])) as math.Matrix);

        return {means: posteriorMean, stds: posteriorStd};
    }

    /**
     * Compute the kernel matrix of two vectors.
     * 
     * @param {number[]} x
     * @param {number[]} y 
     * @return {number[][]} Kernel matrix of a and y
     */
    kernel (x: number[], y: number[]) {
        var kernelMatrix: number[][] = [];
        for (let i = 0; i < x.length; i++) {
            var ithColl: number[] = [];
            for (let j = 0; j < y.length; j++) {
                ithColl.push(this.rbfKernel(x[i], y[j]));
            }
            kernelMatrix.push(ithColl);
        }
        return kernelMatrix;
    }

    /**
     * Compute the RBF kernel of two numbers
     */
    rbfKernel (x: number, y: number) {
        return math.exp(-0.8 * math.square(x - y));
    }
}

export { Optimizer };