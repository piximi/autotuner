import { BaysianOptimisationStep, ModelsDomain, NullableMatrix } from '../types/types';

class Optimizer {
    domainIndices: number[];
    modelsDomains: ModelsDomain;
    nextDomainIndex: number;
    modelsSamples: { [identifier: string]: number[]};
    allSamples: number[];
    observedValues: { [identifier: number]: number};
    best: { index: number, value: number};


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

        if (!this.best || (this.observedValues[this.best.value] < value)) {
            this.best = { index: point, value: value};
        }
    };

    getNextPoint () {
        // if no samples have been added yet (e.g. call 'getNextPoint()' the first time) just pick anyone
        if (this.allSamples.length === 0){
            return { nextPoint: this.domainIndices[0], expectedImprovement: -1}
        }

        const baysianOptimisationStep: BaysianOptimisationStep = { nextPoint: this.domainIndices[this.nextDomainIndex], expectedImprovement: 0};
        this.nextDomainIndex++;
        return baysianOptimisationStep;

    };
}

export { Optimizer };