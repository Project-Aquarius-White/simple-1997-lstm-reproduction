# LSTM 1997 Forward Pass Equations

This document summarizes the mathematical formulations for the Long Short-Term Memory (LSTM) as described in the original 1997 paper (Hochreiter & Schmidhuber).

## Activation Functions

The original architecture uses three distinct squashing functions. The gates use the standard sigmoid, while the cell input and output use scaled and shifted versions to control the range of values.

-   **Gate Sigmoid ($f$):** Used for input and output gates.
    $$f(x) = \sigma(x) = \frac{1}{1 + e^{-x}} \in [0, 1]$$

-   **Cell Input Squashing ($g$):** Squashes the information entering the memory cell.
    $$g(x) = 4\sigma(x) - 2 = \frac{4}{1 + e^{-x}} - 2 \in [-2, 2]$$

-   **Cell Output Squashing ($h$):** Squashes the state before it is gated by the output unit.
    $$h(x) = 2\sigma(x) - 1 = \frac{2}{1 + e^{-x}} - 1 \in [-1, 1]$$

## Forward Pass Equations

For each timestep $t$, the forward pass for a single memory cell $c_j$ is defined by the following equations:

### 1. Input Gate ($y^{in}_j$)
Controls the flow of input signals into the memory cell.
$$net_{in_j}(t) = \sum_u w_{in_j u} y^u(t-1)$$
$$y^{in}_j(t) = f_{in_j}(net_{in_j}(t))$$

### 2. Output Gate ($y^{out}_j$)
Controls the flow of output signals from the memory cell to the rest of the network.
$$net_{out_j}(t) = \sum_u w_{out_j u} y^u(t-1)$$
$$y^{out}_j(t) = f_{out_j}(net_{out_j}(t))$$

### 3. Internal Cell State (CEC update)
The core "Constant Error Carousel" (CEC) update. Note the absence of a forget gate; the previous state $s_{c_j}(t-1)$ is added directly to the gated new input.
$$net_{c_j}(t) = \sum_u w_{c_j u} y^u(t-1)$$
$$s_{c_j}(t) = s_{c_j}(t-1) + y^{in}_j(t) g(net_{c_j}(t))$$

### 4. Cell Output ($y^{c_j}$)
The final hidden state contribution from this memory cell.
$$y^{c_j}(t) = y^{out}_j(t) h(s_{c_j}(t))$$

## Shared Gates (Memory Cell Blocks)

When multiple cells $S$ are grouped into a **Memory Cell Block**, they share the same input and output gates but have individual internal states and input weights:

$$s_{c_j^v}(t) = s_{c_j^v}(t-1) + y^{in}_j(t) g(net_{c_j^v}(t))$$
$$y^{c_j^v}(t) = y^{out}_j(t) h(s_{c_j^v}(t))$$

where $v \in \{1, \dots, S\}$ denotes the cell index within block $j$.

## References
-   **Section 2**: Architecture overview and CEC concept.
-   **Section 3**: Forward pass equations.
-   **Appendix A.1**: Detailed activation function ranges and notation mapping.
