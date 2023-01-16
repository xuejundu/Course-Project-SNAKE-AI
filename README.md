# Beating the Snake Game with Deep Reinforcement Learning

This is a project for our CS486 *Intro to Artificial Intelligence*.
We used TensorFlow to implement the DQN and use Pygame to implement the Snake game.
We adapted the code for the game from [DQN-snake](https://github.com/benjamin-dupuis/DQN-snake) and modified the codes to adapt TensorFlow2.0.
The graphics is adapted from [Project: Train AI to play Snake](https://github.com/maurock/snake-ga).
A refined rewards scheme was written based on [Autonomous agents in Snake game via deep reinforcement learning](https://ieeexplore.ieee.org/document/8460004)

## Requirements

The langage that is used is Python (version 3.5-3.8).

The other libraries used are in the file ```requirements.txt```.

## Usage

Download the necessary libraries:

```
pip install -r requirements.txt
```

To start training a new model or to continue training an existing model, run
```
python train.py --modelName <nameOfYourModel>
```

Arguments can be passed in the previous command to try differents training parameters :

<table>
  <tr>
    <th>Option</th>
    <th>Description</th>
    <th>Default value</th>
    <th>Required</th>
  </tr>
  <tr>
    <td>--modelName</td>
    <td>Name of the model.<br><br>The script would use this model name to save the model, produce output files which record the scores and steps<br><br>Example : --modelName new_model </td>
    <td>---</td>
    <td>Yes</td>
  </tr>
  <tr>
    <td>--learningRate</td>
    <td>Rate at which the agent is learning.<br><br>Example : --learningRate 0.001</td>
    <td>.001</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--memorySize</td>
    <td>Number of events remembered by the agent.<br><br>Example : --memorySize 50000</td>
    <td>100000</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--discountRate</td>
    <td>The discount rate is the<span style="font-weight:bold"> </span>parameter that indicates<br>how<span style="font-weight:bold"> </span>many actions will be considered in the future <br>to evaluate the reward of a given action.  <br>A value of 0 means the agent only <br>considers the present action,<br>and a value close to 1 means the agent<br>considers actions very far in the future.<br><br>Example : --discountRate 0.99</td>
    <td>0.99</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--epsilonMin</td>
    <td>Percentage of random actions selected by the agent.<br><br>Example: --epsilonMin 0.10</td>
    <td>0.00</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--writtenFile</td>
    <td>Specify the name of the output file</td>
    <td>out.txt</td>
    <td>No</td>
  </tr>
  <tr>
    <td>--loadFile</td>
    <td>The model you want to load. If this argument is not specified, a new model would be trained.</td>
    <td></td>
    <td>No</td>
  </tr>
</table>

You should be able to see the training and the models will be saved under models directory.

![](images/demo.gif)

To test the performance of your gamebot, run
```
python test.py --modelName <nameOfYourModel>
```

To play the game yourself, run
```
python play.py
```

Two baseline methods are also implemented, which is random move.

To execute randome_move.py, you need to specify the number of games.
(Replace num_games to any integers)
```
python random_move.py <num_games>
```


## References

- [DQN-snake](https://github.com/benjamin-dupuis/DQN-snake)
- [Project: Train AI to play Snake](https://github.com/maurock/snake-ga)
- [Reinforcement Learning w/ Keras + OpenAI: DQNs](https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c)
- [Building a Powerful DQN in TensorFlow 2.0 (explanation & tutorial)](https://medium.com/analytics-vidhya/building-a-powerful-dqn-in-tensorflow-2-0-explanation-tutorial-d48ea8f3177a)
- [Autonomous agents in Snake game via deep reinforcement learning](https://ieeexplore.ieee.org/document/8460004)

## Contributors

- [Xuejun Du](https://git.uwaterloo.ca/x58du)
- [Xin Wang](https://git.uwaterloo.ca/x772wang)
- [Kyoung Jeon](https://git.uwaterloo.ca/kjjeon)
