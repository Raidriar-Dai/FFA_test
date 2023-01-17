# Report for Implementation on MNIST

## 0.Invariant:
- Manual seed: 3241
- Train_batch_size=50000, Test_batch_size=10000
- num_epochs = 1000
- threshold = 2.0

## 1. Original
- Net: [784, 500, 500]
- Optimizer: Adam
- Learning Rate: 0.03
- Weight decay: 0
- Number of epochs: 1000
- Error:
    - train error: 0.07396000623703003
    - test error: 0.07289999723434448

## 2. Add a layer:
- Net: [784, 500, 500, 500]
- Optimizer: Adam
- Learning Rate: 0.03
- Weight decay: 0
- Number of epochs: 1000
- Error:
    - train error: 0.08066004514694214
    - test error: 0.07990002632141113

## 3. Change the optimizer to AdamW (with default weight_decay argument):
- Net: [784, 500, 500]
- Optimizer: AdamW
- Learning Rate: 0.03
- Weight decay: 0.01
- Number of epochs: 1000
- Error:
    - train error: 0.07488000392913818
    - test error: 0.07330000400543213

## 4. Change the optimizer to SGD:
- Net: [784, 500, 500]
- Optimizer: SGD
- Learning Rate: 0.03
- Weight decay: 0
- Number of epochs: 1000
- Error (**Extremely High**):
    - train error: 0.9146400019526482
    - test error: 0.9161000028252602

## 5. Decrease the learning rate:
- Net: [784, 500, 500]
- Optimizer: Adam
- Learning Rate: 0.003
- Weight decay: 0
- Number of epochs: 1000
- Error (**Relatively High**):
    - train error: 0.4359400272369385
    - test error: 0.4280000329017639

## 6. Increase the number of epochs:
- Net: [784, 500, 500]
- Optimizer: Adam
- Learning Rate: 0.03
- Weight decay: 0
- Number of epochs: 3000
- Error (**Indications of Overfitting**)
    - train error: 0.03266000747680664
    - test error: 0.04630005359649658








