<TeXmacs|1.0.7.16>

<style|generic>

<\body>
  <doc-data|<doc-title|Notes on individual
  project>|<doc-author-data|<author-name|Mihaela Rosca>|<\author-address>
    \;
  </author-address>>>

  <section|Hinton video notes>

  <subsection|Lecture 3>

  Perturbe weights randomly and see which ones improve the error measure that
  we use.

  \;

  Backpropagatation:

  Idea: perturb the activities of the hidden units, instead of the weights,
  because there are less weights\ 

  Hidden units: you do not know what their output should be but you know how
  the error changes as you change the hidden unit.

  Each hidden unit affects multiple output units.

  Idea of back propagation: take the error derivatives wrt to the one later
  of units and propagate them to the layer of units below.

  Wikipedia has a clear explanation on this
  http://en.wikipedia.org/wiki/Backpropagation

  Look especially at the derivation part (should write a bit about the
  history of neural networks and how we got here, so describe a biit the back
  propagation algorithm).

  In algorith, you compute the derrivative of the error with respect to a
  weight and hence once you have the derivative, you can use gradient descent
  to optimize the function

  \;

  Let <math|x<rsub|i> >bt the output of the hidden layer neuron <math|i> and
  <math|net =\<Sigma\>w<rsub|i>*x<rsub|i>> then <math|y>, the output of the
  neuron above is <math|y=\<varphi\><around*|(|net|)>\<nocomma\>>, where
  <math|\<varphi\>> is usually the logistic function. <math|f<around*|(|x|)>=
  <frac|1|1+e<rsup|-x>> and then <frac|\<delta\> f|\<delta\>
  x>=f<around*|(|x|)>*<around*|(|1-f<around*|(|x|)>|)>>

  Then\ 

  <math|<frac|\<delta\> E|d w<rsub|i>>=<frac|\<delta\>E|d*y>*<frac|d y|d
  net>*<frac|d net|d w<rsub|i>>=-<around*|(|t-y|)>*y*<around*|(|1-y|)>*x<rsub|i>>

  \;

  And now you can apply gradient descent to get an update on
  <math|w<rsub|i>>.

  Note that if <math|f<around*|(|x|)>=x> then you get the exact perceptron
  training rule. So this is a generalization of the percetron.

  \;

  Note that the update depend on the training case, so then you have 2
  options:

  <\itemize>
    <item>online: you do it after each training case: a lot of zig zagging
    with the\ 

    <item>full batch: after a full sweep of going trough the data, you
    compute the error wrt to a weight by suming up all individual errors
    obtained from each training set

    <item>mini- bath: after a small sample of training cases: what people do
    when training big networks with a big data set.
  </itemize>

  How much to update the learning rate:

  <\itemize>
    <item>fixed learning rate

    <item>global learning rate

    <item>sepearate learning rate for each weight

    <item>Don't use steepest descent: use different descent directions than
    steepest descent: problem is it is hard to figure out which direction to
    use
  </itemize>

  what we really care about (from an optimization standpoint) is minimizing
  the error with respect to the entire training set, hence the offline
  learning is more stable with respect to the learnign rate.

  <with|font-series|bold|Overfitting>

  Two types of noise

  <\itemize>
    <item>target values are unreliable

    <item>sampling error: accidental regulraies just because of the
    particular training cases that were chosen. A model cannot make the
    difference between the accidental regularities and the real ones.
  </itemize>

  Idea: simpler models are better at generalization! The simpler the model
  the beter. Note that simpler model means both smaller degrees but also
  smaller values for coefficients

  Ways to reduce overfitting

  <\itemize>
    <item>weight decay: keep the weights close to 0

    <item>weight sharing

    <item>early stopping: with test sets, see what happens on the test set,
    and once you get a worse error on a test case you stop
  </itemize>

  <subsection|Lecture 4>

  When encoding inout data as 0,0..01,000 (one 1 per vetor), you are not
  giving false similaritites between the input data to the network: they are
  all as similar between each other.

  \;

  Softmax: forcing the ouput of some neurons to represent a probability
  distribution

  \;

  <\equation*>
    y<rsub|i>=<frac|e<rsup|z<rsub|i>>|<big|sum><rsub|all j>e<rsup|z<rsub|j>>>
  </equation*>

  So the output of unit <math|i> (<math|y<rsub|i>>) does not only depend on
  it's input, but also on the input of the other neurons (<math|z<rsub|j>>)

  Any propbablity disttribution of descrete states can be represented as the
  output of a softmax unit for some inputs.

  The output always lies between (0,1)

  <math|y<rsub|i>> represents the probability of having b

  Using a cost function for the softmax: cross entropy

  <\equation*>
    C = -<big|sum>t<rsub|j>*log y<rsub|j>
  </equation*>

  C has a very big gradient when the target value is 1 and the output is
  almost 0h

  <with|font-series|bold|Speech recongnitoin>

  The standrard triagram method: you compute the relative probabilities of
  words given the previous two words. If you use more than 3 words, you get
  too many zeros, too many sequences that you never have enocountered before.

  Bengio \ neural net for prediction the next word. (maybe mention this in
  the backround research).

  Softmax: are probabilities for each of the possible outputs, so if you have
  a lot of many possible outputs, like 100.000, you need a lot of weights for
  that.

  <with|font-series|bold|Serial architecture>

  word code = feature vector

  <subsection|Lecture 5>

  <with|font-series|bold|Object recognition>

  Why it is hard

  <\itemize>
    <item>lighting: the intenistieis of the pixels are detemined as much by
    the lighting as by the objects

    <item>objects can be deformed in a variaty of non-affine ways

    <item>object classes are often defined by how they are used but rahter of
    how they look like: example given: chairs

    <item>viewpoint: changes in viewpoint changes in images that standard
    learning meathong cannot cope with
  </itemize>

  Dimmension hopping: when the data moves from one dimension to another. You
  need to fix this

  <with|font-series|bold|Achieving viewpoint invariance>:
  <with|font-series|bold|important!!> talk about it in the report and also
  implement it from the project

  Different approaches:

  <\itemize>
    <item>use redundant invariant features

    <item>put a box around the object and use normalized pixels

    <item>use replicated features with pooling: this is called convolutional
    neural nets

    <item>Use a hierarchy of parts that hav explicit poses relative to the
    camera
  </itemize>

  You can put a box around the object so that you determine what the object
  roation is. But it is hard to create the box itself: it is a chicken and
  egg problem.

  The brute force normalization approach: maybe think of this when you are
  stuck?

  <subsection|CONVOLUTIONAL DEEP NETWORKS>

  <with|font-series|bold|in these kind of networks you learn your features.,
  a way to design your features.>

  This is the same as saying as learning an input representation for the
  data.

  \;

  TODO: read more about this

  In computer science, a convolutional neural network is a type of
  feed-forward neural network where the individual neurons are tiled in such
  a way that they respond to overlapping regions in the visual field.

  Convolutional neural networks consist of multiple layers of small neuron
  collections which look at small portions in the input image.

  <with|font-series|bold|You need to normalize the image before>

  The results of these collections are then tiled so that they overlap to
  obtain a better representation of the original image; this is repeated for
  every such layer.

  Because of this, they are able to tolerate translation of the input image.
  Most convolutional networks include local pooling or max pooling layers,
  which simplify and combine the outputs of neighboring neurons; essentially,
  if the outputs are integrated into an image, pooling layers reduce its
  resolution.

  <with|font-series|bold|Convolutional neural networks for hard-written digit
  recognition>

  An example of deep nerual nets.

  Convolutional neural networks are based on replicated features.

  They use many different copies of the same feature detector with different
  positions.

  <with|font-series|bold|Quote from the papers in the 80's about how they did
  it: compare my results with their results and add a nice graph. Try to
  implement the convolution stuff myself, in case I have time!>

  Replication across positions also reduces the number of free parametrs that
  need ot be learned.

  Use several different feature types, each with its own map of replicated
  detectors.

  <with|font-series|bold|Backpropagation for convolution neural networks>

  Here you need to keep some weights equal between each other.

  You start with <math|w<rsub|1>=w<rsub|2>> and you ensure <math|<frac|d E|d
  w<rsub|1>> and <frac|d E|d w<rsub|2>>> that
  <math|\<vartriangle\>w<rsub|1>=\<vartriangle\>w<rsub|2>> , you compute
  <math|<frac|d E|d w<rsub|1>> and <frac|d E|d w<rsub|2>>> and you use their
  sum \ <math|<frac|d E|d w<rsub|1>> + <frac|d E|d w<rsub|2>>> to update both
  the weights, so you keep them equal.

  \;

  What does replication the feature detectors achieve:

  replicated features do not make the neural activities invariant to
  translation., the activities are equivariant.

  <with|font-series|bold|Equivariant in the activties and invariance in the
  weights>

  <with|font-series|bold|Pooling>

  Invariant in the activities:

  Good for face detection, but not really good for facr recognition.

  \;

  <with|font-series|bold|McNemar test>: a good way to compare models, better
  than just comparing the error rates of the 2 models.

  How it works: make a 2 by 2 table:

  <\equation*>
    <tabular|<tformat|<cwith|2|-1|2|-1|cell-halign|c>|<table|<row|<cell|>|<cell|model
    1 wrong>|<cell|model 1 right>>|<row|<cell|model 2
    wrong>|<cell|29>|<cell|1>>|<row|<cell|model 2
    right>|<cell|11>|<cell|9959>>>>>
  </equation*>

  Then you can clearly see which one is better

  Alex used left --right relfectoins of the images to increase the number of
  traning data.

  Dropout stops the network overfitting: omit half of the hidden units in a
  layer are randomly removed for each traning example.

  http://www.youtube.com/watch?v=n6hpQwq7Inw

  Spatial constrast normalization: edge detection: important!!

  <with|font-series|bold|Spatial convolution>

  Convolution also reduces the dimension of the data.

  Tanh and abs increse precision.

  Subsampling

  convolution map

  https://github.com/iskandr/striate for an implementation: o

  <subsection|Lecture 6>

  <with|font-series|bold|Mini batch gradient descent>: less computation than
  with online learning

  Computing the gradient simultaneuosly uses matrix matrix multiplies, which
  are very efficient, especially on GPU

  10 or 100 examples

  minibatched should have balanaced class apparences

  you need a validation set to realise if which learning rate you need to
  use:

  \ \ \ if the errror keeps gettting worse or oscillates widly, reduce the
  learning rate

  \ \ \ 

  towards the end of mini batch learning you need to set the learning rate
  low (when the error stops decreasing).

  <with|font-series|bold|Mini-batch gradient decscent: Tricks>

  <with|font-series|bold|Initializing the weights>

  Initiallize the weights randomly: but proportional to the sqrt of the fan
  in to that weight

  You can also do this for the learning rate.

  \;

  The following ttricks are good for speed:

  <with|font-series|bold|Shift the inputs>: (adding a constant to each of the
  elements of the inputs) substract the <with|font-series|bold|mean> of the
  data from each of the elements. (to make sure the mean is 0)

  <with|font-series|bold|Scaling the input>: scaling the input values such
  that each component of the input vector has unit variance over the whole
  traning set.

  <with|font-series|bold|Better method: decorrelate the input components!!
  Very good method is to implement PCA>

  Take care to not turn down the learning rate to soon

  Four main ways to speed up mini-batch learning

  <\itemize>
    <item>Use momentum

    <item>Use a separate adaptive learning rates for each parameter: look at
    the gradient for that particular parameter. If the gradient fluctuates
    then you decrease the learning rate. If not you can increase the learning
    rate.

    <item>rmspop: Divide the learning rate for a weight by a running average
    of the magnitutes of recent gradients for that weights\ 
  </itemize>

  <with|font-series|bold|Momentum (implus: masa or viteza) method>

  Can be applied to full batch learning but also to mini batch learning.

  We make sure the velocity gets to 0 at some point.

  Equations for the momentum method

  <\equation*>
    v<around*|(|t|)>=*\<alpha\>v<around*|(|t-1|)>-\<varepsilon\><frac|\<delta\>
    E|\<delta\> w><around*|(|t|)>
  </equation*>

  Where <math|t> represents the mini batch iteration update.

  <\equation*>
    \<nabla\>w<around*|(|t|)>=v<around*|(|t|)>=\<alpha\>
    \<nabla\>w<around*|(|t-1|)> -\<varepsilon\><frac|\<delta\> E|\<delta\>
    w><around*|(|t|)>
  </equation*>

  So the current iteration update of the weight depends on previous updates

  <math|\<alpha\>=0.9> usually (has to be close to 1) <math|\<alpha\>> is
  called the momemntum

  So at the beginning is good to have a small momentum (0.5), once the large
  gradients have disappernts and the weights can smoothly raised to 0.9 or
  even 0.99

  <\with|font-series|bold>
    A better type of momentum

    it is better to correct a mistake after you have made it
  </with>

  First make a jump in the direction of the previous accumuldated gradient.
  so I think you increase by <math|\<alpha\> \<nabla\>w<around*|(|t-1|)> and
  >then you compute the gradient at that point
  \ <math|\<varepsilon\><frac|\<delta\> E|\<delta\> w><around*|(|t|)> >with
  the new weights (you need a forward pass for this, to seee what the
  <math|y<rsub|i>> is)

  <with|font-series|bold|A separate adaptive learning rate for each
  connection>

  Use a global learning rate and use a local gain for each connection

  <\equation*>
    \<nabla\>w<rsub|ij>=-\<varepsilon\>g<rsub| i j> <frac|d E|d w<rsub|i j>>
  </equation*>

  <\equation*>
    if <around*|(|<frac|d E|d w<rsub|i j>><around*|(|t|)><frac|d E|d w<rsub|i
    j>><around*|(|t-1|)>|)>\<gtr\>0 then g<rsub|i,j><around*|(|t|)>=g<rsub|i,j><around*|(|t-1|)>+0.05
    else g<rsub|i,j><around*|(|t|)>=g<rsub|i,j><around*|(|t-1|)>*0.95
  </equation*>

  limit the gains into some reasibable range [0.1, 10] or [0.01,10]

  use full batch learning or very big mini batches.

  Adaptive learning rates can be combined with momentum.

  <with|font-series|bold|rmsprop: divide the gradient by a running average of
  its recent magnitute>

  rprop.

  for<with|font-series|bold| full batch learning>, you can only use the sign
  of the gradient (<math|<frac|d E|d w<rsub|i j>>>). all weights updates are
  all of the same maginuted.\ 

  rprop combines this idea with the idea of aadapting the step size
  separately for each step, without looking at the gradient.

  So here you use one adaptive step per weight again:

  <\itemize>
    <item>increase the step size for a weight multiplicatively (times 1.2) if
    the signs of the last 2 gradients agree

    <item>otherwise degrees the step size multiplicatively (times 0.5)

    <item>limit the step size between 50 and 10 ^-6, but I guess it depends a
    lot of the problem. You should onlyuse 50 for problems with the input
    being small\ 
  </itemize>

  rprop does not work really with small mini batches.

  <with|font-series|bold|rmsprop>

  Keep a moving average of the squared gradient for each weight

  <\equation*>
    MeanSquare<around*|(|w,t|)>=0.9*MeanSquare<around*|(|w,t-1|)>+0.1
    <around*|(|<frac|d E|d w><around*|(|t|)>|)><rsup|2>
  </equation*>

  dividing the gradient by <math|<sqrt|MeanSquare<around*|(|w,t|)>>> makes a
  better learning.

  Yann Le Cun `` no more pesky learning rates'' is a new method.

  \;

  <with|font-series|bold|Summary of learning methods for neural networks>

  for small data sets (eg 10.000 cases) or bigger data sets without
  redundancy, use a full batch method

  <\itemize>
    <item>conjugate gradient, LBFG: they come with a package
  </itemize>

  for a big data set you have to use mini bataches

  <\itemize>
    <item>try gradient descent with moment (initially, try without adapting
    learning rates)

    <item>rmsprop (without momentum)

    <item>or try LeCun's latest recipe
  </itemize>

  <subsection|Lecture 7>

  <with|font-series|bold|Modelling sequences>

  we usually want to turn a sequence from one kind to another sequence of
  another sequence (french to english)

  <with|font-series|bold|autoregressive models>

  RNN: non linear! Distirbuted hidden \ states that allows them to store a
  lot of information about the past efficiently.

  RNN: deterministic!

  Can oscilate, and they can settle to point attractors (good for retriving
  memories). They can also behave chaotically.

  Hard to train.

  <with|font-series|bold|Backpropagation trough time>

  RNN is just a feed forward network where the weights are kept constant
  between layers.

  you can also learn the initial states of the of the network.

  How do you specify the input of a RNN:

  1. Specify the initial states of all units.

  2. Specify the initial states of a subset of the units

  3. Specify the states of the same subset of the units at every time step
  (this is the natural way to model most sequential data)

  How to specify the target:

  specify the desired final activities for all the units (all units of one
  step or several few steps - especially good for attractors).

  Rnn's have problems with long range dependencies.

  Four good ways for trainign RNN:

  <\itemize>
    <item>Long short term memory: we change the architecture of the RNN out
    of little moddules that are designed to remember vlaues for a long time

    <item>Hessian free optimization

    <item>Echo State Neurons

    <item>Good initialization with momentum with echo state networks
  </itemize>

  <subsubsection|Long term short term memory>

  short term memory: the dynamic state of a network

  <subsection|Lecture 8>

  It is very useful to see RNN as Feed forward netwroks that have a new layer
  at each time.

  You can use the input to determine the weight connectionvs between the
  layers.

  Remember that with softmax you need to have a small number of possible
  outputs.

  \;

  A lot of parameters: possibility of overfitting

  <with|font-series|bold|IMPORTANT WHEN TALKING ABOUT RNN: they require much
  less traning data to reach the same level of performace than other models>

  Rnn improve faster than other methods as the data sets gets bigger!

  \;

  <with|font-series|bold|Echo state networks>

  Input = state of oscilators

  You can predict the output from the state of the oscilator

  idea: not train hidden to hidden connections, but set them randomly and
  hope you can learn

  can learn sequences of how they affect the output

  <section|Lecture 9>

  How to improve generalization, by reducing overfitting

  How to control the capacity of the netowrk

  How to determine the metaparameters of the network

  Overfitting: any finite set of training data also contains sampling error,
  and there are accidental

  regularities in the traning data. The model it cannot tell which
  regularities are real and which

  are caused by the sampling error

  <\itemize>
    <item>Approach 1: get more data

    <item>Use the model that has the right capacity: very difficult

    <item>Average different models: different forms and different mistakes

    <item>Bayesian: \ find multiple weight vectors that do a good job, and
    average them

    \;
  </itemize>

  <subsubsection|Limit capacity>

  <\itemize>
    <item>Architecure: limit the number of hidden layers and the number of
    units per layer

    <item>Early stopping: start with small eights and stop the learning
    before it overfits

    <item>Weight 0decay: penalize large weights using penalities or
    constrains on their squared values or absolute values

    <item>Add noise to weights or activities

    \;
  </itemize>

  How to set the meta parameters:

  <\itemize>
    <item>Try multiple parameters and see which ones work best for some test
    set

    <item>This makes it unlikely to work it well on another test set

    <item>Choose metaparameters using cross validation: training, validation
    (metaparameters), and test
  </itemize>

  Early stopping:

  start with small weights, and as the performance on the validation set gets
  worse, then you stop the training

  <with|font-series|bold|model with small weights have smaller capcaity>

  <with|font-series|bold|How to limit the size of the weights>

  Introduce a penality that does not let the weights go too large

  add another term for the cost function

  <\equation*>
    C = E+<frac|\<lambda\>|2>*<big|sum><rsub|i>w<rsup|2><rsub|i>
  </equation*>

  <\equation*>
    <frac|d C|d w<rsub|i>>=<frac|d E|d w<rsub|i>>+\<lambda\>*w<rsub|i>
  </equation*>

  Makes the network just use thei weights it needs\ 

  <with|font-series|bold|Weight constrains>

  Better than penalities: just the Larganre multipliers required to keep the
  constraints satisfied

  Instead, we can put a constraint on the maximum squared length of the
  incoming weight vector on each unit.

  If somethign exceeds the maximu, allowed legnth, we scale the weight vector
  down to the scale.

  <\with|font-series|bold>
    Using a noise regularizer
  </with>

  Add either noise to either the weights or the input.

  It is the same as adding a penality on the weights with the L2 norm.

  Add gaussian noise to the input of a neural network. The variance of the
  noise is amplified by the square weight before going to the next layer.

  \;

  <\equation*>
    y<rsub|noisy>=<big|sum><rsub|i>w<rsub|i>*x<rsub|i>+<big|sum><rsub|i>w<rsub|i>*\<varepsilon\><rsub|i>
  </equation*>

  where <math|\<varepsilon\><rsub|i> is sampled from
  N<around*|(|0,\<sigma\><rsub|i><rsup|2>|)>>

  <\equation*>
    E<around*|(|<around*|(|y<rsub|noisy>-t|)><rsup|2>|)>=E<around*|(|<around*|(|y-t+<big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)><rsup|2>|)>
  </equation*>

  <\equation*>
    E<around*|(|<around*|(|y-t+<big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)><rsup|2>|)>=E<around*|(|<around*|(|<around*|(|y-t|)><rsup|2>+<around*|(|<big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)><rsup|2>+2*<around*|(|y-t|)><big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)>|)>
  </equation*>

  <\equation*>
    =E<around*|(|<around*|(|y-t|)><rsup|2>|)>+E<around*|(|<around*|(|<big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)><rsup|2>|)>+2*E<around*|(|<around*|(|y-t|)><big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)>=<around*|(|y-t|)><rsup|2>+E<around*|(|<around*|(|<around*|(|<big|sum>w<rsub|i>*\<varepsilon\><rsub|i>|)><rsup|2>|)>|\<nobracket\>>=<around*|(|y-t|)><rsup|2>+<big|sum>w<rsub|i><rsup|2>*\<sigma\><rsup|2><rsub|i>
  </equation*>

  \;

  Adding noise to the weights works better than addinng noise to the input!

  You can also use noise in the activities of a neuron, by making
  the<with|font-series|bold| units binary and stochastic>

  <math|p<around*|(|s=1|)>=<frac|1|1 \<noplus\>+e<rsup|-z>>>

  <with|font-series|bold|Introduction to the full Bayesian approach>

  Give a probablity distribution for the weights.

  I have a prior probablity distribution of the parameters.

  The prior might be very vague.

  Maximum a posteriory learning

  If you use bayes you get the same weight decay equations

  <\with|font-series|bold>
    minimizing the squared weights is equivalent to maximizing the log
    probablity of the weights under a zero mean guassian prior

    \;
  </with>

  <\equation>
    C = E+<frac|\<sigma\><rsub|<rsup|2>D>|\<sigma\><rsup|2><rsub|W>><big|sum>w<rsup|2><rsub|i>
  </equation>

  This is exactly what we did in MLNC - you can see this in bishop or the
  with the <math|\<alpha\> and \<beta\>>

  <math|\<sigma\><rsub|d>=variance of >the prediction of the model

  <math|\<sigma\><rsub|w>=variance of the prior of the weights>

  \;

  <with|font-series|bold|MacKay's quick and diryt method of fixing weight
  costs>

  does not require a validation set: allows us to have different weight
  penalities for different sets of weights (ie for things like different
  layers)

  the trick is called ``empirical bayes''

  use the Bayesian approach to noise variance and weight variance (eq (1))

  <\itemize>
    <item>set the variance of the gaussian prior to be whatever makes the
    weights that the model learned most likely (you use the data itself to
    decide what your prior is): you try to fit a Gaussian to the data (a
    Gaussian with zero mean). you can learn difference variances from
    different set of weights

    <item>start with guesses for both the noise variance and the weight prior
    variance

    \ \ do gradient descent:

    \ \ \ \ \ do some learning using the ratio of the variances as the weight
    penalty coefficient

    \ \ \ \ \ reset the noise variance to be the variance of the residual
    errors <math|t-y>

    \ \ \ \ set the variance of the weight distribution to be the variance of
    the weights
  </itemize>

  \;

  If you have no useful information in some weights, you expect the error
  derivstive to be small.

  <subsection|Lecture 10>

  good to average model together <math|\<rightarrow\>> better than using an
  individual model

  the more different the models are, the better

  I guess that is why integrating over the weights in the bayes models gives
  best results

  High variance: you fit the sample error of the data

  The more the individual predictors disagree, the better the average
  prediction. Note that this does not mean that you need to have poor
  predictors: on the contrary, y can ou should have the best predictors you
  can

  This generally works if the error is gaussian distrbuted.

  you can try completely different models and see what happnes.

  \;

  for nn:

  <\itemize>
    <item>you can use different number of hidden layers

    <item>different number of units per layer

    <item>different types or strengths of weight penalit
  </itemize>

  You make models differ by changing their trainin data! But it is very
  expensive

  <with|font-series|bold|Mixture of experts>

  Train different neural nets, each for one kind of data.

  You also need a managing input net, that decides to which specialist to
  send the data toh

  \;

  <section|Lecture 11:Hopfield>

  they are deterministic

  Global energy function

  Binary threshold neurons

  we want energy as low as possible. (-E as high as possible)

  <\equation*>
    E=-<big|sum><rsub|i>s<rsub|i>*b<rsub|i>-<big|sum><rsub|i\<less\>j>s<rsub|i>*s<rsub|j>*w<rsub|i,j>
  </equation*>

  You can compute how the value of each neuron affects the energy:

  <\equation*>
    \<nabla\>E<rsub|i>=E<around*|(|s<rsub|i>=0|)>-E<around*|(|s<rsub|i>=1|)>=b<rsub|i>+<big|sum>s<rsub|j>*w<rsub|i,j>
  </equation*>

  And you want to decrease the energy.\ 

  Decisions need to be sequencial.

  You can see local minima as the memory of the network.

  You can put in parts of a pattern and it will converge to a local minima,
  ie one of the stored patterns.

  You can use part recovered patterns, by convering to the local minima - the
  stored patterns in a network.

  Learning in Hopfield:

  <\itemize>
    <item>Using 1 and -1

    \ \ Learning rule <math|\<nabla\>w<rsub|i,j>=s<rsub|i>*s<rsub|j>>

    <item>Using 1 and 0

    \ \ Learning rule <math|\<nabla\>w<rsub|i,j>= 4
    \ <around*|(|s<rsub|i>-<frac|1|2>|)>*<around*|(|s<rsub|j>-<frac|1|2>|)>>
  </itemize>

  Capacity \ of a Hopfield is about 0.15<math|*N> so that you can still
  recover the patterns you have stored in the network, where <math|N> is the
  number of bits in the network (the number of neurons)

  2 local minima can combine to a single local minima, thus runing the cap

  How to get rid off spriupis minima:

  <\itemize>
    <item>unlearning: let the network settle from a random initial state and
    then do unlearning, by appling the opposite of the learning rule

    <item>Different learning rule(made by Elizabeth Gardiner): do not present
    one pattern only once, but many times. Then you use the perceptron
    learning rule to learn the weights of the network, as follows:

    \ \ \ for each unit, you look at what the state of the unit is in case
    you leave all the others like this. If it is the one that you want to
    learn, you leave it like this, otherwise you add the input(which is the
    other neurons in the network) to all the weights, or substract -like in
    the perceptron learning procedure. this is guaranteed to converge if such
    a set of weights exists

    mention this in the report!!
  </itemize>

  <with|font-series|bold|Hidden units for Hopfield>

  add hidden units that are good interpretation of the input data.

  Idea: instead of using the net to store memories, use it to construct
  interpretation of the sensory input.

  The input is represented by the visible units.

  The interpretation is represented by the states of the hidden units.

  The energy of the system will represent the badness of the representation
  given by the input units.

  <with|font-series|bold|HIDDEN UNITS = representation of the input>

  <with|font-series|bold|Using stochastic units to improve search>

  search = the hidden units described above, get stuck in a local minima of
  the energy function

  for simple Hopfield, it is impossible to get out of a local minima, as you
  never go up in energy

  solution: add random noise

  Idea: use random noise to escape local minima. start with a lot of noise so
  its easy to cross energy bariers. Then you slowly decrease the noise
  -simulated anealing

  replace the binary threshold units by binary stochastic units that make
  biased random decisions.

  <\equation*>
    p<around*|(|s<rsub|i>=1|)>=<frac|1|1+e<rsup|-\<nabla\>E<rsub|i>/T>>
  </equation*>

  where <math|T> is the temperature.

  Raising the noise level is equivalent to decreasing all the energy gaps
  between configurations (ie the change in energy when one neuron goes from
  one state to the other)

  <math|\<nabla\>E<rsub|i>> is the energy gap.

  Note that if the <math|T=0\<nocomma\>> the sign of <math|\<nabla\>E<rsub|i>
  >deterimes the probability, to be either 0 or 1, making the decision unit
  to be deterministic.

  If <math|T=1> you just have the logistic function

  <with|font-series|bold|Thermal equilibruim>

  Does not mean that the system settles down into the lowest energy
  configuration.

  What settles down is the probablity distribution over configurations: this
  settles down to the stationary distribution.

  How to think about thermal equilibrium: \ imagine a huge ensamble of
  systems that have the exact same energy function (and same weights).

  We can define the probability of a configuration as the fraction of systems
  that are in the same configuration.

  Thermal equilibruim: the fraction of systems in any configuration in any
  particular configuration does not change. This is guaranteed to happen if
  the weights of the network are symmetric.

  Note that so far we have talked about Stochastic Hopfield with hidden units
  (those are Boltzmann machines)

  <with|font-series|bold|How a Botlzmann machine models data>

  Given a training set of binary vectors, fit a model that will assign a
  probablity to every possible binary vector:

  \ \ \ this is useful for deciding if other binary vectors come from the
  same distribution (eg. documents represented by binary features that
  represents the occurence of a particular world)

  \ \ you can have different kinds of distributions, and you want to see to
  which one the vector fits best: it allows you to compute the posterior
  probabilities to determine which model determined this data sample (under
  the assumption that the data was generated under one of your models)

  <\equation*>
    p<around*|(|model<rsub|i><around*|\||data|\<nobracket\>>|)>=<frac|p<around*|(|data<around*|\||model<rsub|i>|\<nobracket\>>|)>*p<around*|(|model<rsub|i>|)>|p<around*|(|data|)>>=<frac|p<around*|(|data<around*|\||model<rsub|i>|\<nobracket\>>|)>*p<around*|(|model<rsub|i>|)>|<big|sum>p<around*|(|data<around*|\||model<rsub|j>|)>p<around*|(|model<rsub|i>|)>*|\<nobracket\>>>
  </equation*>

  \;

  1. Casual generative model

  Hidden units: latent variables.

  Step1: you determine the states of the hidden units by sampling them from
  latent variables

  Step2: Uses weights and biases to determine the probability of a visible
  vector given the hidden state

  <\equation*>
    p<around*|(|v|)>=<big|sum><rsub|h>p<around*|(|v<around*|\||h|\<nobracket\>>|)>*p<around*|(|h|)>
  </equation*>

  2. Energy based model

  not a causal generative model

  Two ways of defining the joint probability

  <\itemize>
    <item><math|p<around*|(|v,h|)>\<propto\>e<rsup|-E<around*|(|v,h|)>>>\ 

    <item>The probability that the network is in a state with <math|v> and
    <math|h> after we have updated the network enough times such that we have
    reached a thermal equilibruim
  </itemize>

  Turns out that the above 2 are equivalent

  <\equation*>
    p<around*|(|v,h|)>=<frac|e<rsup|-E<around*|(|v,h|)>>|<big|sum><rsub|u,g><rsup|>e<rsup|-E<around*|(|v,h|)>>>
  </equation*>

  <with|font-series|bold|partition function>
  <math|<big|sum><rsub|u,g><rsup|>e<rsup|-E<around*|(|v,h|)>>>

  Problem, partition function is too big, cannot be computed!

  So we use Markov Chain Monte Carlo to get samples from the model starting
  with a random global configuration until you reach termal equilibruim, and
  then you see what values of <math|v> and <math|h> and you know that
  <math|p<around*|(|v,h|)>\<propto\>e<rsup|-E<around*|(|v,h|)>>>

  \;

  You can also sample for the posterior hidden distributions, in the same
  way, by keeping the visible units constants (only update the hidden units)

  \;

  <with|font-series|bold|Boltzmann machine learning>

  The goal of the learning (no backpropagation, there are no labels, this is
  not supervised learning)

  We want to maximize the product of the probabilities that the Botlzmann
  machine assigns to the binary vectors in the training set \ 

  You can do this with the following learning algorithm, tht only requires
  local information

  <\equation*>
    <frac|\<delta\> log p<around*|(|v|)>|\<delta\>
    w<rsub|i,j>>=\<less\>s<rsub|i>*s<rsub|j>\<gtr\><rsub|v>-\<less\>s<rsub|i>*s<rsub|j>\<gtr\><rsub|model>
  </equation*>

  <math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\>v> is the expected value of product
  of states at thermal equilibruim given that <math|v> is clamped in the
  visible units

  <math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\><rsub|model>> the product of the
  same states at thermal equilibruim, with nothing clamped

  <math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\>v> = first time in learning in
  Hopfield net

  <math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\><rsub|model>> = how to get rid of
  sprious minima in Hopfield

  Ways of sampling from the model of the RBM:\ 

  <math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\>v> decreases the energy
  <math|E<around*|(|v,h|)>> so increases <math|e<rsup|-E<around*|(|v,h|)>>>
  for some <math|h> after we have clamped v

  <math|><math|\<less\>s<rsub|i>*s<rsub|j>\<gtr\><rsub|model>> decreases the
  partition function, by looking at points with low energy

  <with|font-series|bold|More efficient ways to get the statistics for
  training the Boltzmann>

  no easy test to see if we have reached thermal equilibruim\ 

  why not start from whatever state you ended up last time you saw that data
  vector? This stored state 'the interpretation of the given data vector' is
  called a <with|font-shape|italic|particle>.

  Using particles that persist to get a 'warm start' (ie they have been
  obtained after running the network for some time) has a big advantage: if
  we were at equilibuim the last time and we only changes the weights a
  little, then we just have to run the network for a bit to get again to the
  equilibruim.

  <with|font-series|bold|Neal's method for collecting statistics>

  <\itemize>
    <item>positive phase: Keep a set of data specific particles (hidden
    states when the data has been clamped) and use them to train. You keep
    keep one or more particles for each data vector. Each particle has a
    current value that is a configuration of the hidden units. You
    sequentially update all the hidden units a few times in each particle
    with the relevant data vector clamped. For every pair of connected units,
    you compute <math|s<rsub|i>*s<rsub|j> >and then you average it over all
    particles (here you can already see that this is not online, but rather
    batch learning. Does not work well with mini-batches)

    <item>negative \ phase: Keep a set of ``fantasy particles'' (they have a
    global configuration, nothing is clamped). In an update you
    <with|font-shape|italic|sequentially> update all units in each fantasy
    particle a few times (you update the visible units as well). For every
    connected pair units average <math|s<rsub|i>*s<rsub|j>> over all the
    fantasy particles.
  </itemize>

  This works better for full batch learning
  <math|\<nabla\>w<rsub|i,j>=\<less\>s<rsub|i>,s<rsub|j>\<gtr\><rsub|v>-\<less\>s<rsub|i>,s<rsub|j>\<gtr\>
  <rsub|model>>

  This does not work well with mini-bathces because the next time you see a
  particle vector for a data instance, you have already modified the weights
  multiple times (for the other batches)

  You can overcome this by assuming that when e data vector is clamped, the
  set of good hidden units that explain the vector is unimodal (we only have
  one good explanaition for our data vector) and that there is only one
  hidden state that gives a local minima for energy for each data vector.

  <with|font-series|bold|The mean field approximation>

  only works with the uni model assumption

  we have stochastic and sequential update:

  <\equation*>
    p<around*|(|s<rsub|i>=1|)>=\<sigma\><around*|(|b<rsub|i>+<big|sum>s<rsub|j>*w<rsub|i,j>|)>
  </equation*>

  where <math|\<sigma\>> is the logistic sigmoid.

  Instead of doing that we can use the probablities themselves

  <\equation*>
    p<rsup|t+1><rsub|i>=\<sigma\><around*|(|b<rsub|i>+<big|sum>p<rsup|t><rsub|j>*w<rsub|i,j>|)>
  </equation*>

  This is not techniqally correct, due to the fact that <math|\<sigma\>> is
  non linear. Now you can update all units in parallel.

  This can lead to oscialltations, and to avoid that we can use the
  <with|font-series|bold|dumped mean field:>

  <\equation*>
    p<rsup|t+1><rsub|i>=\<lambda\>*p<rsup|t><rsub|i>
    +<around*|(|1-\<lambda\>|)>*\<sigma\><around*|(|b<rsub|i>+<big|sum>p<rsup|t><rsub|j>*w<rsub|i,j>|)>
  </equation*>

  <with|font-series|bold|How to use mini batch learning for BM>

  <\itemize>
    <item>positive phase: initialize all hidden probabilities to 0.5. Clamp
    the visible unit vectors to the data vector. Keep computing the
    probabilities <math|p<rsub|i> for> each hidden state (in parallel) until
    you get convergence (note that now you can check convergence using
    <math|p<rsub|i>>, before you could not really check convergence due to
    the stochastic nature of the problem, while the probabilities themselves
    are not stochastic)

    <item>Negative phase: same as before, keep a set of phantasy particles,
    each representing a global configuration. And you sequentially update the
    units in each fantasy particle a few times. For every connected pair of
    units average <math|\<less\>s<rsub|i>,s<rsub|j>>\<gtr\> over all fantasy
    particles.

    <item>The difference between the two averages is what you use for
    updating the weights
  </itemize>

  Deep boltzman machine: no connections between neurons of the same layer.
  Update in parallel the even layers and then the odd layers.

  You cannot see the sampling and the learnig and the sampling as two
  different processes. The learning is affecting the sampling.

  <with|font-series|bold|Restricted bolztmann machines and their learning>

  no connection between hidden units: one layer of hidden units and no
  connections between visible units

  In an RBM it only takes one step to reach thermal equilibruim with the
  visible units clamped, so we can easily compute the value of
  <math|\<less\>v<rsub|i>,h<rsub|j>\<gtr\><rsub|v>>

  <\equation*>
    p<around*|(|h<rsub|j>=1|)>=<frac|1|1+e<rsup|-<around*|(|b<rsub|j>+<big|sum>v<rsub|i>*w<rsub|i,j>|)>>>
  </equation*>

  <with|font-series|bold|Note that they are independent of each other so we
  can compute them in parallel!!>

  <with|font-series|bold|PCD: >an efficient mini-batch learning procedure on
  RBM

  <\itemize>
    <item>positive phase: clamp <math|v> on the visible units. Compute
    \<less\><math|v*<rsub|i>h<rsub|j>\<gtr\>> for all possible hidden units.
    You can do this exactly because you can compute the exact probability of
    <math|h<rsub|j> and v<rsub|i> are> fixed. Note that in this case it is
    just <math|v<rsub|i>*p<around*|(|h<rsub|j>=1|)>\<nosymbol\>>. For every
    connected pair of units, average <math|\<less\>v<rsub|i>,h<rsub|j>\<gtr\>>over
    all the data in the mini batch.

    <item>negative phase: keep a set of ``fantasy particles'' (nothing is
    clamped). Each particle has it's own global configuration. Update each
    particle a few times using alternative parllel updates (update the hidden
    states in parallel, and update the visible states in parallel). Then
    average <math|v<rsub|i>h<rsub|j>> over all fantasi parrticles.
  </itemize>

  Works quite well and allows RBMs to build good density models.

  <with|font-series|bold|Contrastive divergence>; can you repeat the same
  test data?

  Instead of doing multiple steps, you just do 2 (this is for CD1)

  <\itemize>
    <item>build a vector of hidden units from the data (you can do this in
    parallel)

    <item>reconstruct the visible units from the hidden vector (you can do
    this in parallel)

    <item>build the vector of hidden units from the reconstruction

    <item><math|\<nabla\>w<rsub|i,j>=\<varepsilon\><around*|(|\<less\>v<rsub|i>*h<rsub|j>\<gtr\><rsub|0>-\<less\>v<rsub|i>h<rsub|j>\<gtr\><rsub|1>|)>>
  </itemize>

  Note that <math|\<varepsilon\>> here is a step size so all the things from
  the lectures before apply (check that to see how to adapt).

  Once the weights are big enough you use CD3, CD10 etc.

  <with|font-series|bold|An example of RBM learning>

  works for hand written digits.

  data: 16 x 16 pixels (but how do you convert to binary?)

  50 hidden units (feature detectors)

  <with|font-series|bold|you start with small random weights: DIFFERENT!>

  Uses CD1 to learn

  <\itemize>
    <item>increase the weights when a visible unit and a feature unit are
    active at the same time in the first step

    <item>decrease the weights when a visible unit and a feature unit are
    active at the same time in the second step
  </itemize>

  Note that this is slightly different than CD, where you decrease the
  expected values!

  You can visualize the each of the weights to a feature detector as a 16 x
  16 image (can be super useful to see what the features represent)

  How to reconstruct digits: ie you pass it via the network once

  each neuron: feature

  \;

  <with|font-series|bold|Rbm for collaborative filtering>

  You can use visible units to be 5 ways softmaxs intead of value.

  About 100 hidden units.

  The CD learning for the softmax is the same as for a binary unit?

  you use an RBM for each user: visible units: as many movies as he rated\ 

  all RBMS will share weights: if two users rated the same movie, the weights
  from the visible unit of this user to the hidden units will be the same for
  all users

  You average the predicitions of the RBM with matrix factorization models

  \;

  <with|font-series|bold|How to choose how many hidden units to use?>

  <section|From Andrew Ng's talk>

  http://www.youtube.com/watch?v=n1ViNeWhC24

  The idea is to find a better way to represent the data than just the raw
  data

  <with|font-series|bold|Self taought learning>

  You give a set of unlabeled images (random images) and learn a much better
  representation of images than the raw pixels and then you can use a small
  traning set to build a classifier.

  instead of giving motercycles and not motorcycles and building a
  classification.\ 

  When the brain sees something, the first thing it does it looks for
  <with|font-series|bold|edges>.

  Sparse coding

  First layer: pixel -\<gtr\> edges

  Second layer: edges -\<gtr\> object parts

  Third layer: object parts -\<gtr\> object models

  <section|Suggestions>

  Think of using GPU for traning to speed up

  write about the two versions of PCA which are implemented and the
  comparison between them.

  <with|font-series|bold|You need to normalize the image before>: increases
  the training speed.

  <with|font-series|bold|Convolution>

  \;

  the amount of overalap of one function as it is shifted over another
  function

  Talk about convolution neural nets and the difference between it and
  stacked RBMs

  \;

  <with|font-series|bold|Make a chapter about RNN.>\ 

  <with|font-series|bold|IMPORTANT WHEN TALKING ABOUT RNN: they require much
  less traning data to reach the same level of performace than other models>

  Talk about the string models in chapter 8, trained to learn from Wikipedia.

  Read from Tomas Mikolov(Lecture 8.3) and mention he is the state of the art
  model of traning RNN to predict the next word (after traning from
  wikipedia).

  Rnn improve faster than other methods as the data sets gets bigger!

  When talking about RBM talk about all the implementations that could be,
  and then the difference in speed in training. Implement multiple of them
  and show how the change in time versus precision
</body>

<\initial>
  <\collection>
    <associate|language|american>
    <associate|page-type|letter>
  </collection>
</initial>

<\references>
  <\collection>
    <associate|auto-1|<tuple|1|?>>
    <associate|auto-10|<tuple|2|?>>
    <associate|auto-11|<tuple|2.0.1|?>>
    <associate|auto-12|<tuple|2.1|?>>
    <associate|auto-13|<tuple|3|?>>
    <associate|auto-14|<tuple|4|?>>
    <associate|auto-15|<tuple|5|?>>
    <associate|auto-16|<tuple|6|?>>
    <associate|auto-2|<tuple|1.1|?>>
    <associate|auto-3|<tuple|1.2|?>>
    <associate|auto-4|<tuple|1.3|?>>
    <associate|auto-5|<tuple|1.4|?>>
    <associate|auto-6|<tuple|1.5|?>>
    <associate|auto-7|<tuple|1.6|?>>
    <associate|auto-8|<tuple|1.6.1|?>>
    <associate|auto-9|<tuple|1.7|?>>
  </collection>
</references>

<\auxiliary>
  <\collection>
    <\associate|toc>
      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Hinton
      video notes> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-1><vspace|0.5fn>

      <with|par-left|<quote|1.5fn>|Lecture 3
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-2>>

      <with|par-left|<quote|1.5fn>|Lecture 4
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-3>>

      <with|par-left|<quote|1.5fn>|Lecture 5
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-4>>

      <with|par-left|<quote|1.5fn>|CONVOLUTIONAL DEEP NETWORKS
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-5>>

      <with|par-left|<quote|1.5fn>|Lecture 6
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-6>>

      <with|par-left|<quote|1.5fn>|Lecture 7
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-7>>

      <with|par-left|<quote|3fn>|Long term short term memory
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-8>>

      <with|par-left|<quote|1.5fn>|Lecture 8
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-9>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Lecture
      9> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-10><vspace|0.5fn>

      <with|par-left|<quote|3fn>|Limit capacity
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-11>>

      <with|par-left|<quote|1.5fn>|Lecture 10
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-12>>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|From
      Andrew Ng's talk> <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-13><vspace|0.5fn>

      <vspace*|1fn><with|font-series|<quote|bold>|math-font-series|<quote|bold>|Suggestions>
      <datoms|<macro|x|<repeat|<arg|x>|<with|font-series|medium|<with|font-size|1|<space|0.2fn>.<space|0.2fn>>>>>|<htab|5mm>>
      <no-break><pageref|auto-14><vspace|0.5fn>
    </associate>
  </collection>
</auxiliary>