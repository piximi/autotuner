# autotuner
Hyperparameter search module for tensorflow.js (sequential) models.  
The following parameters are autotuned: lossFunction, optimizer algorithm, batchSize and epochs

# Usage

```javascript
import { TensorflowlModelAutotuner } from '@piximi/autotuner';
```

### Getting Started

Initialize the autotuner by specifying metrics, a dataset and the number of categories.
```javascript
var autotuner = new TensorflowlModelAutotuner(['accuracy'], dataset, numberOfCategories);
```
### Adding a model to the autotuner
```javascript
// create some uncompiled sequential tensorflow model
const testModel = await createModel();

const parameters = { lossfunction: [LossFunction.categoricalCrossentropy], optimizerAlgorithm: [tensorflow.train.adadelta()], batchSize: [10], epochs: [5,10] };
autotuner.addModel('testModel', testModel, parameters);
```

### Tuning the hyperparameters
Specify the optimization algorithm: the hyperparameters can be tuned by either doing bayesian optimization or by doing a simple grid search.
```javascript
autotuner.bayesianOptimization();
```
```javascript
autotuner.gridSearchOptimizytion();
```
The ojective function of the optimization can be specified (either 'error' or 'accuracy'):
```javascript
autotuner.gridSearchOptimizytion('accuracy');
```
Evaluating a model can be done using cross validation:
```javascript
autotuner.gridSearchOptimizytion('accuracy', true);
```
When doing bayesian optimization the maximum number of domain points to be evaluated can be specified as an optional parameter:
```javascript
autotuner.bayesianOptimization('accuracy', true, 0.8);
```
In the example above the optimizytion stops after 80% of the domain ponits have been evaluated. By default this value is set to 0.75.  
### Evaluate the best hyperparameters
The best hyperparameters found in the optimization run can be evaluated on the test set. Specify the objective and wheter or not to use cross validation.
```javascript
autotuner.evaluateBestParameter('error', true);
```

### Example
An example usage can be found here:
```bash
tets/runExampleAutotuner.ts
```
# Development

Pull and initialize:
```bash
git clone https://github.com/piximi/autotuner.git
cd autotuner
npm install
```

To run tests:
```bash
npm run test
npm run runExampleAutotuner
```

To compile the code and check for type errors:
```bash
npm run build
```