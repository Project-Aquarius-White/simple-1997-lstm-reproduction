import numpy as np
from math_helpers import *
class Layers():
    class LSTMMemoryCell():
        def __init__(self, input_size, hidden_size, initialization = "xavier", activation = "tanh"):
            self.hidden_size = hidden_size
            stacked_size = input_size+hidden_size

            if activation == "tanh":
                self.activation = modified_tanh
                self.activation_derivative = tanh_derivative
            else : 
                self.activation = modified_sigmoid
                self.activation_derivative= modified_sigmoid_derivative

            #Weight Initilization
            if initialization == "xavier": initialize = np.sqrt(1.0/input_size)
            else :initialize = 0.001
            self.input_weights = np.random.randn(stacked_size,hidden_size) * initialize
            self.input_gate_weights = np.random.randn(stacked_size, hidden_size) * initialize
            self.output_gate_weights = np.random.randn(stacked_size, hidden_size) * initialize

            #Biases Initialization
            self.input_biases = np.zeros(hidden_size)
            self.input_gate_biases = np.full(hidden_size, -2.0)
            self.output_gate_biases = np.full(hidden_size, -1.0)

            #State_Cell_Initialization
            self.state_cell = np.zeros(hidden_size)
            self.state_cell_history = []
            self.i = 0
            
            pass
        
        def cell_forward(self, stacked_input:np.ndarray, t= 0) -> np.ndarray:
            input_signal = self.activation(np.dot(stacked_input, self.input_weights) + self.input_biases)
            input_gate_activation = sigmoid(np.dot(stacked_input, self.input_gate_weights) +self.input_gate_biases)
        
            self.state_cell += input_signal * input_gate_activation
            self.state_cell_history.append(self.state_cell.copy())

            output_signal = self.activation(self.state_cell)
            output_gate_activation = sigmoid(np.dot(stacked_input, self.output_gate_weights) +self.output_gate_biases)

            output_activation = output_signal*output_gate_activation

            math_cache = {
                "input_signal" : input_signal,
                "input_gate_activation": input_gate_activation,
                "output_signal": output_signal,
                "output_gate_activation": output_gate_activation
            }

            return output_activation, math_cache
        
        def cell_backward(self, stacked_input, cache, dh_next, state_cell_next, time_step):
            input_signal = cache["input_signal"]
            input_gate_activation = cache["input_gate_activation"]
            output_signal = cache["output_signal"]
            output_gate_activation = cache["output_gate_activation"]
            state_cell_t = self.state_cell_history[time_step]

            d_output_gate = dh_next*sigmoid_derivative(output_gate_activation)*self.activation(state_cell_t)
            d_output_gate_weights = np.dot(stacked_input.T, d_output_gate)
            d_output_gate_biases = np.sum(d_output_gate, axis=0)

            d_state_cell = dh_next * output_gate_activation * self.activation_derivative(output_signal) + state_cell_next

            d_input_gate = d_state_cell*sigmoid_derivative(input_gate_activation)*input_signal 
            d_input_gate_weights = np.dot(stacked_input.T, d_input_gate)
            d_input_gate_biases = np.sum(d_input_gate, axis=0)

            d_input = d_state_cell*tanh_derivative(input_signal)*input_gate_activation
            d_input_weights = np.dot(stacked_input.T, d_input)
            d_input_biases = np.sum(d_input, axis=0)

            d_input_activation = np.dot(d_input_gate, self.input_gate_weights.T) + \
                     np.dot(d_input, self.input_weights.T) + \
                     np.dot(d_output_gate, self.output_gate_weights.T)


            return d_input_gate_weights, d_input_gate_biases, d_input_weights, d_input_biases, d_output_gate_weights, d_output_gate_biases, d_input_activation, d_state_cell
            
        def update_cell_weights_biases(self, GA_weights, GA_biases, learning_rate):
            self.input_weights -= learning_rate*GA_weights["input_weight_gradients"]
            self.input_biases -= learning_rate*GA_biases["input_bias_gradients"]

            self.input_gate_weights -= learning_rate*GA_weights["input_gate_weight_gradients"]
            self.input_gate_biases -= learning_rate*GA_biases["input_gate_bias_gradients"]

            self.output_gate_weights -= learning_rate*GA_weights["output_gate_weight_gradients"]
            self.output_gate_biases -= learning_rate*GA_biases["output_gate_bias_gradients"]

        def reset_state(self, batch_size=1):
            self.state_cell = np.zeros((batch_size, self.hidden_size)) 
            self.state_cell_history = []