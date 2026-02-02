# import RNN
import numpy as np
from math_helpers import *
from layers import Layers
from optimizer import Optimizers
class MewralOldLSTM():
    def __init__(self, input_size, hidden_size, no_of_hidden_layers, output_size, optimizer="sgd_momentum", initialization = "xavier"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_layer_count = no_of_hidden_layers
        self.Hidden_Layers = []

        prev_layer_size = input_size
        for _ in range(no_of_hidden_layers):
            self.Hidden_Layers.append(Layers.LSTMMemoryCell(prev_layer_size, hidden_size))
            prev_layer_size = hidden_size
        
        self.Weights_hy = np.random.randn(hidden_size, output_size) * np.sqrt(1.0/self.hidden_size)
        self.Biases_hy = np.zeros(output_size)
        self.momentum_coef = 0.9
        if optimizer == "sgd_momentum":
            self.optimizer:Optimizers.SGDwMomentum = Optimizers.SGDwMomentum(self.Hidden_Layers, self.Weights_hy, self.Biases_hy, lr = 0.05, mu=self.momentum_coef)
        elif optimizer == "sgd":
            self.optimizer:Optimizers.SGD = Optimizers.SGD(self.Hidden_Layers, self.Weights_hy, self.Biases_hy, lr = 0.05)

    def BPTT(self, inputs, targets, learning_rate=0.05, loss_mode="last_step", retain_state = False): 
        # Forward Pass
        self.optimizer.learning_rate = learning_rate
        outputs, hidden, caches = self.forward(inputs, retain_state=retain_state)
        
        # Loss Calculation for Logging
        if loss_mode == "sequence":
            loss = 0.0
            error_list = []
            for t in range(len(outputs)):
                err = outputs[t] - targets[t]
                loss += 0.5 * np.sum(err**2)
        elif loss_mode == "last_step":
            final_error = outputs[-1] - targets[-1]
            loss = 0.5 * np.sum(final_error**2)

        # Backward Pass
        GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h = self.backward(inputs, outputs, targets, hidden, caches, loss_mode)

        # Norm Clipping
        max_grad_norm = 0.5
        total_norm = 0.0
        
        # Calculate Norm
        total_norm += np.sum(GA_weights_hy ** 2)
        total_norm += np.sum(GA_bias_y ** 2)
        for layer_grads in GA_hidden_hh:
            for key in layer_grads:
                total_norm += np.sum(layer_grads[key] ** 2)
        for layer_grads in GA_bias_h:
            for key in layer_grads:
                total_norm += np.sum(layer_grads[key] ** 2)
        
        total_norm = np.sqrt(total_norm)

        self.last_batch_short_term_memory = None
        
        clip_coef = 1.0
        if total_norm > max_grad_norm:
            clip_coef = max_grad_norm / (total_norm + 1e-6)

        self.Weights_hy, self.Biases_hy = self.optimizer.optimize(
            GA_weights_hy, 
            GA_bias_y, 
            GA_hidden_hh, 
            GA_bias_h, 
            self.Hidden_Layers, 
            self.Weights_hy, 
            self.Biases_hy, 
            clip_coef
        )

        return loss  

    def forward(self,inputs, retain_state = False):

        batch_size = inputs.shape[1]

        
        first_layer_shape = self.Hidden_Layers[0].state_cell.shape
        shape_mismatch = (len(first_layer_shape) < 2) or (first_layer_shape[0] != batch_size)

        if retain_state == False or shape_mismatch:
            for layer in self.Hidden_Layers:
                layer.reset_state(batch_size)
        hidden = {}

        if retain_state and self.last_batch_short_term_memory is not None:
             hidden[-1] = self.last_batch_short_term_memory
        else:
             hidden[-1] = np.zeros((self.hidden_layer_count, batch_size, self.hidden_size))

        outputs = []

        caches = []
        for t in range(len(inputs)):
            
            x_t = inputs[t]
            prev_time_hidden_layers = hidden[t-1]
            current_input = x_t

            current_hidden = []
            i=0
            layer_math_cache = []
            layer: Layers.LSTMMemoryCell
            for layer in self.Hidden_Layers:
                stack = np.concatenate((current_input, prev_time_hidden_layers[i]), axis=1)
                layer_out, cache = layer.cell_forward(stack, t)
                layer_math_cache.append(cache)
                current_hidden.append(layer_out)
                current_input = layer_out
                i+=1
            caches.append(layer_math_cache)
              
            hidden[t] = np.array(current_hidden)
            y_t = np.dot(current_hidden[-1], self.Weights_hy) + self.Biases_hy
            outputs.append(y_t)
            
        self.last_batch_short_term_memory = hidden[len(inputs)-1]
        return outputs, hidden, caches
    
    def backward(self, inputs, outputs, targets, hidden, caches, loss_mode="last_step"):
        # GA Initialization for output layer weights and Biases
        GA_weights_hy = np.zeros_like(self.Weights_hy)
        GA_bias_y = np.zeros_like(self.Biases_hy)
        batch_size = inputs.shape[1]

        GA_hidden_hh  = [{"input_weight_gradients": np.zeros_like(x.input_weights), 
                          "input_gate_weight_gradients": np.zeros_like(x.input_gate_weights), 
                          "output_gate_weight_gradients": np.zeros_like(x.output_gate_weights)} 
                          for x in self.Hidden_Layers]
        
        GA_bias_h  = [{"input_bias_gradients": np.zeros_like(x.input_biases), 
                       "input_gate_bias_gradients": np.zeros_like(x.input_gate_biases), 
                       "output_gate_bias_gradients":np.zeros_like(x.output_gate_biases)} 
                       for x in self.Hidden_Layers]

        dh_next = np.zeros((self.hidden_layer_count, batch_size, self.hidden_size))
        state_cell_next = np.zeros((self.hidden_layer_count, batch_size, self.hidden_size))
        # T = len(inputs)
        for t in reversed(range(len(inputs))):
        
            if loss_mode == "sequence":
                loss_derivative = outputs[t] - targets[t]
            else:
                if t == len(inputs) - 1:
                    loss_derivative = outputs[t] - targets[t]
                else:
                    loss_derivative = np.zeros_like(outputs[t])
            # loss_derivative /= len(inputs)

            current_time_hidden = hidden[t]
            GA_weights_hy += np.dot(current_time_hidden[-1].T, loss_derivative)
            GA_bias_y += np.sum(loss_derivative, axis = 0)
            
            layer:Layers.LSTMMemoryCell
            current_layer_grad = np.dot(loss_derivative, self.Weights_hy.T)
            
            for layer, index in zip(reversed(self.Hidden_Layers), reversed(range(self.hidden_layer_count))):
                total_dh = current_layer_grad + dh_next[index]
                if t > 0:
                    h_prev_t = hidden[t-1][index]
                else:
                    h_prev_t = np.zeros((batch_size, self.hidden_size))

                if index == 0:
                    layer_input = inputs[t]
                    split_point = self.input_size
                else:
                    layer_input = hidden[t][index-1]
                    split_point = self.hidden_size

                stack = np.concatenate((layer_input, h_prev_t), axis=1)
                d_input_gate_weights, d_input_gate_biases, d_input_weights, d_input_biases, \
                d_output_gate_weights, d_output_gate_biases, d_input_activation, d_state_cell = \
                        layer.cell_backward(stack, caches[t][index],total_dh,state_cell_next[index], t)
                GA_hidden_hh[index]["input_weight_gradients"] += d_input_weights
                GA_hidden_hh[index]["input_gate_weight_gradients"] += d_input_gate_weights
                GA_hidden_hh[index]["output_gate_weight_gradients"] += d_output_gate_weights
                GA_bias_h[index]["input_bias_gradients"] += d_input_biases
                GA_bias_h[index]["input_gate_bias_gradients"] += d_input_gate_biases
                GA_bias_h[index]["output_gate_bias_gradients"] += d_output_gate_biases
                state_cell_next[index]= d_state_cell 

                current_layer_grad = d_input_activation[:, :split_point]
                dh_next[index]= d_input_activation[:, split_point:]

        return GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h
                


