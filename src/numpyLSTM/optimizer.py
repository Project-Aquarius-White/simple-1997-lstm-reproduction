from layers import Layers
import numpy as np


# These Optimizers only work with the LSTM Structure and wont work on any other Type of model
class Optimizers:
    class SGD:
        def __init__(self, Hidden_Layers, Weights_hy, Biases_hy, lr, mu=0.9):
            self.learning_rate = lr
            self.momentum_coeff = mu
            self.Hidden_Layers = Hidden_Layers

        def optimize(self, GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h, Hidden_Layers, weights_hy, bias_y, clip_coef):
            new_weights_hy = weights_hy - self.learning_rate * (GA_weights_hy * clip_coef)

            new_biases_hy = bias_y - self.learning_rate * (GA_bias_y * clip_coef)

                    
            layer:Layers.LSTMMemoryCell
            for index, layer in enumerate(Hidden_Layers):
                # Scaled Gradients
                scaled_weights = {k: v * clip_coef for k, v in GA_hidden_hh[index].items()}
                scaled_biases = {k: v * clip_coef for k, v in GA_bias_h[index].items()}
                
                layer.input_weights -=self.learning_rate * scaled_weights["input_weight_gradients"]
                
                layer.input_gate_weights -= self.learning_rate * scaled_weights["input_gate_weight_gradients"]

                layer.output_gate_weights -=self.learning_rate * scaled_weights["output_gate_weight_gradients"]

                layer.input_biases -= self.learning_rate * scaled_biases["input_bias_gradients"]

                layer.input_gate_biases -=self.learning_rate * scaled_biases["input_gate_bias_gradients"]

                layer.output_gate_biases -=self.learning_rate * scaled_biases["output_gate_bias_gradients"]

            return new_weights_hy, new_biases_hy    

        pass

    class SGDwMomentum:
        def __init__(self, Hidden_Layers, Weights_hy, Biases_hy, lr, mu=0.9):
            self.learning_rate = lr
            self.momentum_coeff = mu
            self.Hidden_Layers = Hidden_Layers

            self.v_Weights_hy = np.zeros_like(Weights_hy)
            self.v_Biases_hy = np.zeros_like(Biases_hy)
            self.v_layers = []
            l:Layers.LSTMMemoryCell
            for l in self.Hidden_Layers:
                self.v_layers.append({
                    "input_weights": np.zeros_like(l.input_weights),
                    "input_gate_weights": np.zeros_like(l.input_gate_weights),
                    "output_gate_weights": np.zeros_like(l.output_gate_weights),
                    "input_biases": np.zeros_like(l.input_biases),
                    "input_gate_biases": np.zeros_like(l.input_gate_biases),
                    "output_gate_biases": np.zeros_like(l.output_gate_biases)
                })
            pass
        
        def optimize(self, GA_weights_hy, GA_bias_y, GA_hidden_hh, GA_bias_h, Hidden_Layers, weights_hy, bias_y, clip_coef):
            self.v_Weights_hy = self.momentum_coeff * self.v_Weights_hy - self.learning_rate * (GA_weights_hy * clip_coef)
            new_weights_hy = weights_hy + self.v_Weights_hy 
            
            self.v_Biases_hy = self.momentum_coeff * self.v_Biases_hy - self.learning_rate * (GA_bias_y * clip_coef)
            new_biases_hy = bias_y + self.v_Biases_hy

                    
            layer:Layers.LSTMMemoryCell
            for index, layer in enumerate(Hidden_Layers):
                # Scaled Gradients
                scaled_weights = {k: v * clip_coef for k, v in GA_hidden_hh[index].items()}
                scaled_biases = {k: v * clip_coef for k, v in GA_bias_h[index].items()}
                
                # Momentum Update for Layer
                v = self.v_layers[index]
                
                v["input_weights"] = self.momentum_coeff * v["input_weights"] - self.learning_rate * scaled_weights["input_weight_gradients"]
                layer.input_weights += v["input_weights"]
                
                v["input_gate_weights"] = self.momentum_coeff * v["input_gate_weights"] - self.learning_rate * scaled_weights["input_gate_weight_gradients"]
                layer.input_gate_weights += v["input_gate_weights"]
                
                v["output_gate_weights"] = self.momentum_coeff * v["output_gate_weights"] - self.learning_rate * scaled_weights["output_gate_weight_gradients"]
                layer.output_gate_weights += v["output_gate_weights"]
                
                v["input_biases"] = self.momentum_coeff * v["input_biases"] - self.learning_rate * scaled_biases["input_bias_gradients"]
                layer.input_biases += v["input_biases"]
                
                v["input_gate_biases"] = self.momentum_coeff * v["input_gate_biases"] - self.learning_rate * scaled_biases["input_gate_bias_gradients"]
                layer.input_gate_biases += v["input_gate_biases"]
                
                v["output_gate_biases"] = self.momentum_coeff * v["output_gate_biases"] - self.learning_rate * scaled_biases["output_gate_bias_gradients"]
                layer.output_gate_biases += v["output_gate_biases"]

            return new_weights_hy, new_biases_hy    