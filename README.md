# autotuner
Hyperparameter search module for tensorflow.js (sequential) models.  
The following parameters are autotuned: lossFunction, optimizer algorithm, batchSize and epochs

# Usage

```javascript
import { TensorflowlModelAutotuner } from '@piximi/autotuner';
```

### Getting Started

Initialize the autotuner by specifying metrics and a dataset.
```javascript
var autotuner = new TensorflowlModelAutotuner(['accuracy'], dataset.trainData, dataset.testData, dataset.validationData);
```
### Adding a model to the autotuner
```javascript
// create some uncompiled sequential tensorflow model
const testModel = await createModel();

const parameters = { lossfunction: [LossFunction.categoricalCrossentropy], optimizerAlgorithm: [tensorflow.train.adadelta()], batchSize: [10], epochs: [5,10] };
autotuner.addModel('testModel', testModel, parameters);
```

### Tuning the hyperparameters
Specify the optimization algorith. The hyperparameters can be tuned by either doing bayesian optimization or by doing a simple grid search. 
```javascript
autotuner.tuneHyperparameters("bayesian");
```
```javascript
autotuner.tuneHyperparameters("gridSearch");
```
The autotuner can reuse the observations collected on a previous optimization run.
```javascript
autotuner.tuneHyperparameters("bayesian", true);
```

An example usage can be found here:
```bash
tets/runExampleAutotuner.ts
```
# Development

Pull and initialize:
```bash
git pull https://github.com/piximi/autotuner.git
cd autotuner
npm install
```

To run tests:
```bash
npm test
```

To compile the code and check for type errors:
```bash
npm build
```