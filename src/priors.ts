import equal from 'deep-equal';
import * as math from 'mathjs';

class Priors {
    domain: number[];
    observedValues: any;
    // TODO: type mean and kernel properly
    mean: any;
    kernel: any;

    constructor (domain: number[]) {
    this.domain = domain
    this.observedValues = {}

    var domainSize: number = Object.keys(this.domain).length + 1;
    for (var i=0; i < domainSize; i++) {
        this.observedValues[domain[i]] = [];
    }
    this.mean = math.zeros(domainSize);
    this.kernel = math.identity(domainSize);
    }

    // TODO: create type for observedValues
    commit (observedValues: any) {
        var domainSize: number = Object.keys(this.domain).length + 1;

        for (var point in observedValues) {
            this.observedValues[point].push(observedValues[point]);

            // Find domain index.
            var idx = this.domain.findIndex((x: number) => equal(x, point));

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
            if (this.observedValues[this.domain[i]].length === 0) {
                this.mean[i] = sum / count;
            }
        }

        // Recompute the kernel by using the standard covariance function between all observed points.
        for (var point in observedValues) {
            var idx = this.domain.findIndex((x: number) => equal(x, point));
            for (var point2 in observedValues) {
                if (this.observedValues[point2].length > 0){
                    var idx2 = this.domain.findIndex((x: number) => equal(x, point2));
                    var cov = 0.0;
                    for (var i = 0; i < this.observedValues[point].length; i++) {
                        for (var j = 0; j < this.observedValues[point2].length; j++) {
                            if (i <= j) {
                                cov += (this.observedValues[point][i] - this.mean[idx]) * (this.observedValues[point2][j] - this.mean[idx2]);
                            }
                        }
                    }
                    cov /= (this.observedValues[point].length * this.observedValues[point2].length)
                    this.kernel[idx][idx2] = cov;
                    this.kernel[idx2][idx] = cov;
                }
            }
        }
    };
}

export { Priors };