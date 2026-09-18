
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

before that we need to understand 4 concepts

1. generalization
2. regularization
3. overfitting
4. underfitting


Generalization means - How well the trained model performs on new/unseen data.

Overfitting:
Training performance → excellent
Unseen/test performance → poor

Underfitting:
Training performance → poor
Unseen/test performance → poor

Regularization refers to techniques that reduce overfitting and improve generalization.
eg: include weight decay, dropout, data augmentation, and early stopping



- 2 questions we generally get
    1. what is optimiser?
    2. what are different optimiser?


## what is optimiser?

in backpropagation we calculate gradient for model parameters.
this gradient tell us How does the loss change if this parameter changes?
but loss.backward() -> doesn't update the weights.
why do we need to update the weights?
because we want the loss to be minimal.
so we need optimiser

-> w = w - learning_rate * (w.grad)
this is a basic gradient descent.

so now typical pytorch code will be 
optimiser.zero_grad()
predictions = model(X)
loss = loss_fn(predictions,y)
loss.backwards()
optimiser.step()

zero_grad() clears previously accumulated gradients, backward() calculates new gradients, and step() uses those gradients to update parameters.

## now what are different types of optimisers?

#### 1 epoch = training full dataset one time. 
GD -> SGD -> Mini-batch -> Momentum -> RMSProp -> Adam -> AdamW

suppose we have 10k samples

GD:
use entire dataset to calculate one gradient before updating the parameter.
but this not useful right. it's fine 10k , but not sutiable for large datasets.

SGD:
SGD uses one training example at a time.
for 10k samples it is 10k epochs
Updates are cheap but very noisy.

Mini-batch:
we train in batches. 
batch_size = 32
so 10k/32 -> 313 batches -> 313 updates per epoch

Momentum:
SGD only considers the current gradient.
It doesn't remember previous gradients.
This can cause oscillation(means front i.e + and back i.e - ) and slow movement through certain loss landscapes
Momentum idea is simple Remember previous gradient direction and build velocity.
new_m = beta * m + (1-beta)*g

in pytorch:
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9
)

RMSProp:
Using one global learning rate can produce overly large steps in steep directions and tiny steps in shallow directions(horizantal direction)
v = beta * v + (1-beta)*g^2 - Maintain an exponential moving average of squared gradients
 so w = w - learning_rate * (g/(sqrt(v)+epsilon))

RMSProp = adapt step size based on recent gradient magnitude
Historically large gradients → larger denominator → smaller effective step.
Historically small gradients → smaller denominator → relatively larger effective step


Adam:
combines both Momentum and RMSProp
we add some bias also for momentum and rmsprop. 
because when we are inital stages of training.. the values of both is towards zero. 
and it takes time to get what it is initally.. 

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


AdamW -> Adam + Weight decay 

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=0.001,
    weight_decay=0.01
)

Most important:
loss.backward() = calculate gradients
optimizer.step() = use gradients to update parameters




