import { BaysianOptimisationStep, ModelsDomain, NullableMatrix } from '../types/types';

class Optimizer {
    domainIndices: number[];
    modelsDomains: ModelsDomain;
    nextDomainIndex: number;
    modelsSamples: { [identifier: string]: number[]};
    allSamples: number[];
    observedValues: { [identifier: number]: number};

    constructor (domainIndices: number[], modelsDomains: ModelsDomain, mean: NullableMatrix = null, kernel: NullableMatrix = null) {
        this.domainIndices = domainIndices;
        this.modelsDomains = modelsDomains;
        this.nextDomainIndex = 0;

        // contains dictionary that maps from model to sampled data points
        this.modelsSamples = {};
        for (var model in modelsDomains) {
            this.modelsSamples[model] = [];
        }
        this.allSamples = [];
        this.observedValues = {};
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
        const baysianOptimisationStep: BaysianOptimisationStep = { nextPoint: this.domainIndices[this.nextDomainIndex], acquisitionFunctionValue: 0};
        this.nextDomainIndex++;
        return baysianOptimisationStep;

    };
}

export { Optimizer };