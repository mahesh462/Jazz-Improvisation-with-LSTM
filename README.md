# Jazz-Improvisation-with-LSTM
Application of LSTM for music generation.

# Problem statement
We will train a network to generate novel jazz solos in a style representative of a body of performed work.

# Music as Sequence of Values
- We will obtain a dataset of values, and will learn an RNN model to generate sequences of values.
- This music generation system will use 78 unique values.

X:  (m,  T<sub>x</sub> , 78) dimensional array.
- We have m training examples, each of which is a snippet of  Tx=30  musical values.
- At each time step, the input is one of 78 different possible values, represented as a one-hot vector.
- For example, X[i,t:] is a one-hot vector representing the value of the i-th example at time t.


Y: a  (T<sub>y</sub>,m,78)  dimensional array.
- This is essentially the same as X, but shifted one step to the left (to the past).
- Notice that the data in Y is reordered to be dimension  (T<sub>y</sub>,m,78) , where  T<sub>y</sub>=T<sub>x</sub> . This format makes it more convenient to feed into the LSTM later.
- So our sequence model will try to predict  y<sup>⟨t⟩</sup> given x<sup>⟨1⟩</sup>,…,x<sup>⟨t⟩</sup>. 

n_values: The number of unique values in this dataset. This should be 78.


indices_values: python dictionary mapping integers 0 through 77 to musical values.


## Overview of the Model
Here is the architecture of the model we will use:
<p align = 'center'>
  <img src = '/images/music_generation.png'>
</p>

- X=(x<sup>⟨1⟩</sup>,…,x<sup>⟨Tx⟩</sup>)  is a window of size Tx scanned over the musical corpus.
- Each  x<sup>⟨t⟩</sup>  is an index corresponding to a value.
- <a href="https://www.codecogs.com/eqnedit.php?latex=\hat{y^{t}}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\hat{y^{t}}" title="\hat{y^{t}}" /></a>   is the prediction for the next value.
- We will be training the model on random snippets of 30 values taken from a much longer piece of music.
  - Thus, we won't bother to set the first input  x<sup>⟨1⟩</sup> = <a href="https://www.codecogs.com/eqnedit.php?latex=\underset{0}{\rightarrow}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\underset{0}{\rightarrow}" title="\underset{0}{\rightarrow}" /></a>  , since most of these snippets of audio start somewhere in the middle of a piece of music.
  - We are setting each of the snippets to have the same length  T<sub>x</sub>=30  to make vectorization easier.
 
 
- We will train a model that predicts the next note in a style that is similar to the jazz music that it's trained on. The training is contained in the weights and biases of the model.
- We're then going to use those weights and biases in a new model which predicts a series of notes, using the previous note to predict the next note.
- The weights and biases are transferred to the new model using 'global shared layers'.

## Building the Model
- If you're building an RNN where, at test time, the entire input sequence  x<sup>⟨1⟩</sup>,x<sup>⟨2⟩</sup>,…,x<sup>⟨Tx⟩</sup>  is given in advance, then Keras has simple built-in functions to build the model.
- However, for sequence generation, at test time we don't know all the values of  x<sup>⟨t⟩</sup>  in advance.
- Instead we generate them one at a time using  x<sup>⟨t⟩</sup>=y<sup>⟨t−1⟩</sup>.
  - The input at time "t" is the prediction at the previous time step "t-1".

## Shareable Weights
- The function djmodel() will call the LSTM layer  T<sub>x</sub>  times using a for-loop.
- It is important that all  Tx  copies have the same weights.
  - The  Tx  steps should have shared weights that aren't re-initialized.
- Referencing a globally defined shared layer will utilize the same layer-object instance at each time step.
- The key steps for implementing layers with shareable weights in Keras are:
  - Define the layer objects (we will use global variables for this).
  - Call these objects when propagating the input.
# Generating Music
We now have a trained model which has learned the patterns of the jazz soloist. Lets now use this model to synthesize new music.

## Predicting & Sampling
<p align = 'center'>
  <img src = '/images/music_gen.png'>
</p>

At each step of sampling, we will:
- Take as input the activation 'a' and cell state 'c' from the previous state of the LSTM.
- Forward propagate by one step.
- Get a new output activation as well as cell state.
- The new activation 'a' can then be used to generate the output using the fully connected layer, densor.
