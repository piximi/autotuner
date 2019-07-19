import * as tensorflow from '@tensorflow/tfjs';
import { StringModelParameters, ModelMapping, ModelsDomain, Domain } from '../types/types';

class Paramspace {
    // 'models': mapping from a model to the parameters
    models: ModelMapping;

    // 'domain': array of type {'model' : 'model1', 'params' : {'a' : 1, 'b' : 10}} where 'a' and 'b' are parameters
    // i.e. all the 'points' from where we are searching for the best
    domain: Domain[];

    // 'domainIndices': array of domain indices
    domainIndices: number[];
    
    // 'modelsDomains': mapping from the model identifier to the indices of the models parameters in the domain
    modelsDomains: ModelsDomain;

    constructor() {
        this.models = {};
        this.domain = [];
        this.domainIndices = [];
        this.modelsDomains = {};
    }

    addlModel (modelIdentifier: string, modelParameters: StringModelParameters) {
        // Add model to model collection.
        this.models[modelIdentifier] = modelParameters;
        
        // Expand model parameters.
        var modelDomain: StringModelParameters[]= [modelParameters];
        var newElements = false;
        do {
            var params = modelDomain.shift();
            for (var key in params) {
                if (params.stringParameters[key].constructor === Array) {
                    for (var i = 0; i < params.stringParameters[key].length; i++) {
                        var p = JSON.parse(JSON.stringify(params));
                        p[key] = params.stringParameters[key][i];
                        modelDomain.push(p);
                    }
                    newElements = true;
                    break;
                } else {
                    newElements = false;
                }
            }
            if (!newElements) {
                if (params) {
                    modelDomain.unshift(params);
                }
            }
        } while(newElements);

        // Add indices of the model's domain section to the modelDomains object.
        this.modelsDomains[modelIdentifier] = Array.from(new Array(modelDomain.length), (x,i) => i + Object.keys(this.domain).length);

        // Extend the domain with new points defined by the model name and parameters.
        this.domain = this.domain.concat(Array.from(modelDomain, (p) => {return {'model' : modelIdentifier, 'params' : p}}) as any);

        // Create a list of domain indices. We can use them instead of object for faster operations.
        this.domainIndices = Array.from(this.domain, (x,i) => i);
    };
}

export { Paramspace };