import { StringParameters, ModelMapping, ModelsDomain, Domain } from '../types/types';

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

    addModel (modelIdentifier: string, modelParameters: StringParameters) {
        // Add model to model collection.
        this.models[modelIdentifier] = modelParameters;
        
        var parameterValues = Object.values(modelParameters);
        var parameterNames = Object.keys(modelParameters);
        var numberOfParameters = parameterValues.length;

        var ParametersCartesianProduct: number[][] = [[]];
        for (let i = 0; i < numberOfParameters; i++) {
            let currentSubArray = parameterValues[i];
            let temp = [];
            for (let j = 0; j < ParametersCartesianProduct.length; j++) {
              for (let k = 0; k < currentSubArray.length; k++) {
                temp.push(ParametersCartesianProduct[j].concat(currentSubArray[k]));
              }
            }
            ParametersCartesianProduct = temp;
        }

        var modelDomain: [{ [name: string]: number}] = [{}];
        for (let i = 0; i < ParametersCartesianProduct.length; i++) {
            let tempModelParameter: { [parameterName: string]: number} = {};
            for (let j = 0; j < numberOfParameters; j++) {
                //Array.from(modelParameters, (p) => {return {'model' : modelIdentifier, 'params' : p}}) as any);
                tempModelParameter[parameterNames[j]] = ParametersCartesianProduct[i][j];
            }
            modelDomain.push(tempModelParameter);
        }

        // Add indices of the model's domain section to the modelDomains object.
        var domainLength = Object.keys(this.domain).length;
        this.modelsDomains[modelIdentifier] = Array.from(new Array(modelDomain.length), (x,i) => i + domainLength);

        // Extend the domain with new points defined by the model name and parameters.
        this.domain = this.domain.concat(Array.from(modelDomain, (p) => {return {'model' : modelIdentifier, 'params' : p}}) as any);

        // Create a list of domain indices. We can use them instead of object for faster operations.
        this.domainIndices = Array.from(this.domain, (x,i) => i);
    };
}

export { Paramspace };