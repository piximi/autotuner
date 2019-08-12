import * as math from 'mathjs';
import { BayesianOptimizer, Paramspace, Priors } from '../src/index';

describe('Optimizer', () => {
    it('should be initialized properly', () => {
        var domainIndices = [1,2,3,4,5];
        var modelsDomains = {'a' : [1,2], 'b' : [3,4,5]};
        var mean = math.ones(domainIndices.length) as math.Matrix;
        var kernel = math.ones(domainIndices.length, domainIndices.length) as math.Matrix;
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains, mean, kernel);
        expect(optimizer.domainIndices).toEqual([1,2,3,4,5])
        expect(optimizer.modelsDomains['a']).toEqual([1,2])
        expect(optimizer.modelsDomains['b']).toEqual([3,4,5])
    });

    it('should be able to add a sample', () => {
        var domainIndices = [1,2,3,4,5];
        var modelsDomains = {'a' : [0,1,2,3,4]};
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains);
        optimizer.addSample(2, 1.0);
        expect(optimizer.observedValues[2]).toEqual(1.0);
    });
    
    it('should compute the next point', () => {
        var domainIndices = [1,2];
        var modelsDomains = {'a' : [0,1]};
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains);
        optimizer.addSample(2, 1.0);
        var point = optimizer.getNextPoint();
        expect(point.nextPoint).toEqual(1);
    });

    it('should compute the next point after 3 samples', () => {
        var domainIndices = [1,2,3,4,5];
        var modelsDomains = {'a' : [0,1,2,3,4]};
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains);
        optimizer.addSample(2, 1.0);
        optimizer.addSample(1, 2.0);
        optimizer.addSample(4, 0.5);
        var point = optimizer.getNextPoint();
        expect(point.nextPoint).not.toBe(1);
        expect(point.nextPoint).not.toBe(2);
        expect(point.nextPoint).not.toBe(4);
    });

    it('should take the mean prior into account when computing the next point', () => {
        var domainIndices = [1,2,3,4,5];
        var modelsDomains = {'a' : [0,1,2,3,4]};
        var mean = math.matrix([0, 0, 0, 0, 3]);
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains, mean=mean);
        optimizer.addSample(2, 1.0);
        optimizer.addSample(1, 2.0);
        optimizer.addSample(4, 2.5);
        var point = optimizer.getNextPoint();
        expect(point.nextPoint).toEqual(5);
    });

    it('should take the correlated points into account when computing the next point', () => {
        var domainIndices = [1,2,3,4];
        var modelsDomains = {'a' : [0,1,2,3]};
        var kernel = math.matrix([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1]
        ]);
        var optimizer = new BayesianOptimizer.Optimizer(domainIndices, modelsDomains, null, kernel);
        optimizer.addSample(2, 1.0);
        optimizer.addSample(4, 2.5);
        var point = optimizer.getNextPoint();
        expect(point.nextPoint).toEqual(3);
    });
});



describe('Paramspace', () => {

    it ('should be initialized properly', () => {
        var result = new Paramspace.Paramspace();
    });

    it ('should expand a single parameter array', () => {
        var p = new Paramspace.Paramspace();
        p.addModel('model1', {'a' : [1,2,3], 'b' : [10]});
        expect(p.domain).toEqual([
            {'model' : 'model1', 'params' : {'a' : 1, 'b' : 10}}, 
            {'model' : 'model1', 'params' : {'a' : 2, 'b' : 10}}, 
            {'model' : 'model1', 'params' : {'a' : 3, 'b' : 10}}
        ]);
    });

    it ('should expand two parameter arrays', () => {
        var p = new Paramspace.Paramspace();
        p.addModel('model1', {'a' : [1,2], 'b' : [3,6]});
        expect(p.domain).toEqual([
            {'model' : 'model1', 'params' : {'a' : 1, 'b' : 3}},
            {'model' : 'model1', 'params' : {'a' : 1, 'b' : 6}},
            {'model' : 'model1', 'params' : {'a' : 2, 'b' : 3}},
            {'model' : 'model1', 'params' : {'a' : 2, 'b' : 6}}
        ]);
    });

    it ('should expand two parameter arrays where one is a single element array', () => {
        var p = new Paramspace.Paramspace();
        p.addModel('model1', {'a' : [1,2], 'b' : [3]});
        expect(p.domain).toEqual([
            {'model' : 'model1', 'params' : {'a' : 1, 'b' : 3}},
            {'model' : 'model1', 'params' : {'a' : 2, 'b' : 3}},
        ]);
    });

    it ('should assign proper indices to models', () => {
        var p = new Paramspace.Paramspace();
        p.addModel('model1', {'a' : [1,2], 'b' : [3]});
        p.addModel('model2', {'a' : [1,2], 'b' : [3]});
        expect(p.modelsDomains['model1']).toEqual([0, 1]);
        expect(p.modelsDomains['model2']).toEqual([2, 3]);
    });
});

describe('Priors', () => {

    it ('should be initialized properly', () => {
        var domainIndices = [1,2,3];
        var priors = new Priors.Priors(domainIndices);
        expect(priors.mean).toEqual(math.matrix([0,0,0]));
        expect(priors.kernel).toEqual(math.matrix([[1,0,0], [0,1,0], [0,0,1]]));
    });

    it ('should compute the mean', () => {
        var domainIndices = [1,2,3];
        var priors = new Priors.Priors(domainIndices);
        priors.commit({1:2});
        priors.commit({1:4, 2:6});
        expect(priors.mean).toEqual(math.matrix([3,6,4]));
    });

});
