import { BaysianOptimisationStep } from '../types/types';

class Optimizer {
    domainIndices: number[];
    nextDomainIndex = 0;

    constructor (domainIndices: number[]) {
        this.domainIndices = domainIndices;
    }

    addSample (point: number, value: number) {
    };

    getNextPoint () {
        const baysianOptimisationStep: BaysianOptimisationStep = { nextPoint: this.domainIndices[this.nextDomainIndex], acquisitionFunctionValue: 0};
        this.nextDomainIndex++;
        return baysianOptimisationStep;
    };
}

export { Optimizer };