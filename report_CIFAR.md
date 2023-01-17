# Report for Implementation on CIFAR10

## 0. Invariant
- Manual seed: 1234
- Train_batch_size=5000, Test_batch_size=10000
- num_epochs = 1000
- threshold = 2.0

## 1. No Dropout
- AdamW (weight_decay = 0.01);
- No dropout;
- net = Net ([3072, 1000, 1000])
- Error:
    - Average Train Error: 0.19098002314567566
    - Final Train Error: 0.5823000073432922
    - Test Error: 0.644899994134903

## 2. With Dropout, and deeper network
- AdamW (weight_decay = 0.01);
- With Dropout(p = 0.25);
- net = Net([3072, 1000, 1000, 1000])
- Error:
    - Average Train Error: 0.215580016374588
    - Final Train Error: 0.5821000039577484
    - Test Error: 0.6471000015735626

## 3. With Dropout, and smaller network
- AdamW (weight_decay = 0.01);
- With Dropout(p = 0.25);
- net = Net([3072, 500, 500])
- Error:
    - Average Train Error: 0.28556001782417295
    - Final Train Error: 0.5981000065803528
    - Test Error: 0.6480000019073486

## 4. With Dropout, deeper network, higher weight_decay and default learning rate
- AdamW (weight_decay = 0.05);
- With Dropout(p = 0.25);
- net = Net([3072, 1000, 1000, 1000]);
- Error:
    - Average Train Error: 0.23500002026557923
    - Final Train Error: 0.6278000175952911
    - Test Error: 0.6781000196933746

## 5. With Dropout, deeper network, higher weight_decay and lower learning rate
- AdamW (weight_decay = 0.05);
- With Dropout(p = 0.25);
- net = Net([3072, 1000, 1000, 1000]);
- lr = 0.02 (Default lr = 0.03, for all other tests)
- Error:
    - Average Train Error: 0.25824001133441926
    - Final Train Error: 0.6202000081539154
    - Test Error: 0.6739000082015991

