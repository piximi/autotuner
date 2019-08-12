import * as math from 'mathjs';
import { equal } from './util';

class Priors {
    domainIndices: number[];
    // 'observedValues': mapping from a point to its observed values
    observedValues: { [identifier: string]: number[]};
    mean: number[];
    kernel: math.Matrix;

    constructor (domainIndices: number[]) {
        this.domainIndices = domainIndices
        this.observedValues = {}

        var domainSize: number = Object.keys(this.domainIndices).length;
        for (var i=0; i < domainSize; i++) {
            this.observedValues[domainIndices[i]] = [];
        }
        this.mean = math.zeros(domainSize) as number[];
        this.kernel = math.identity(domainSize) as math.Matrix;
    }

    commit (observedValues: {[identifier: number]: number}) {
        var domainSize: number = Object.keys(this.domainIndices).length;

        for (var point in observedValues) {
            this.observedValues[point].push(observedValues[point]);

            // Find domain index.
            var idx = this.domainIndices.findIndex((x: number) => equal(x, point));

            // Recompute the mean.
            this.mean[idx] = math.mean(this.observedValues[point]);
        }

        // We find the points that have never been sampled and assign them with the mean taken over the whole sample set.
        var sum = 0;
        var count = 0;
        for (var point in this.observedValues) {
            if (this.observedValues[point].length > 0) {
                    sum += this.observedValues[point].reduce((a: any,b: any) => a+b);
                    count += this.observedValues[point].length;
            }
        }
        for (var i = 0; i < domainSize; i++) {
            if (this.observedValues[this.domainIndices[i]].length === 0) {
                this.mean[i] = sum / count;
            }
        }

        // Recompute the kernel by using the standard covariance function between all observed points.
        for (let point in observedValues) {
            var idx: number = this.domainIndices.findIndex((x: number) => equal(x, point)) + 1;
            for (let point2 in observedValues) {
                if (this.observedValues[point2].length > 0){
                    var idx2: number = this.domainIndices.findIndex((x: number) => equal(x, point2)) + 1;
                    var cov: number = 0.0;
                    for (let i: number = 0; i < this.observedValues[point].length; i++) {
                        for (let j: number = 0; j < this.observedValues[point2].length; j++) {
                            if (i <= j) {
                                cov += (this.observedValues[point][i] - this.mean[idx]) * (this.observedValues[point2][j] - this.mean[idx2]);
                            }
                        }
                    }
                    // TODO: test matrix operations and casts to matrix
                    cov /= (this.observedValues[point].length * this.observedValues[point2].length)
                    this.kernel = math.subset(this.kernel, math.index(idx, idx2), cov) as math.Matrix;
                    this.kernel = math.subset(this.kernel, math.index(idx2, idx), cov) as math.Matrix;
                }
            }
        }
    };
}

export { Priors };