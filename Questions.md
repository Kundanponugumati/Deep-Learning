
why different loss functions
why differnt optimizers

# steps in neural networks 

# 1. forward pass

model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,1)
)
prediction = model(X)

# 2. to calculate loss

for that we have many functions
we basically divide into 2.
- 1. regression
- 2. classification

regression.
we have 
1. MSE (Mean Square Error) - nn.MSELoss()
we punishes large mistakes aggresively
not suitable when we have outliers

2. MAE (Mean Absoulute Error) - nn.L1Loss()
here we are less sensitive to outliers.

3. Huber Loss - nn.HuberLoss(delta=1.0)
- here we choose a delta 
and when error is small i.e <= delta -> MSE
and when error is big i.e > delta -> MAE

classification

here we having 2 losses.

1. for only 2 output chances. like either 0/1
we use BCE(Binary Cross Entropy) - nn.BCELoss()
here Loss L = -[y*logp +(1-y)*log(1-p)]

what ever we get from model(X) -> we call them as ** logit **

logit = model(X)
loss_fn = nn.BCELogitsLoss()
loss = loss_fn(logit,y)

2. for multi class outputs we have Cross Entropy
so here our NN will predict many outputs 
suppose we take it as 3. 
so now our model is like 
model = nn.Sequential(
    nn.Linear(2,4),
    nn.ReLU(),
    nn.Linear(4,3)
)
logits = model(X) -> output is of shape (3,)

now if we add them we wont get 1 . 
to make it probabilities we use softmax function. e^x/(sigma(e^x))

similary here also we can calculate loss 
loss_fn = nn.CrossEntropy()
loss = loss_fn(logits,y)


# 3. BackPropagation
we do backpropagation. 
will do using chain rule. all the mathematical stuff. 
- we use loss.backward()

# 4. Optimiser.step()
will see optimiser tmrw


