"""
Shared color constants for LSTM thesis defense animations.

This module defines a consistent color scheme for visualizing LSTM components
and dynamics in ManimGL animations. Colors are carefully chosen to represent
different gates, states, and data flows in the LSTM architecture.

Color Scheme:
- Cell State (CEC): Blue (#3B82F6) - The constant error carousel
- Input Gate: Green (#22C55E) - Controls input information
- Output Gate: Orange (#F97316) - Controls output information
- Cell Input (g): Teal (#14B8A6) - Raw input transformation
- Gradient Flow: Red (#EF4444) - Backpropagation signals
- Weights/Connections: Gray (#6B7280) - Network connections
- Background: Dark (#1E1E2E) - Scene background
- Text: White (#FFFFFF) - Labels and annotations
- Accent/Markers: Yellow (#FACC15) - Highlights and markers

Usage:
    from assets.animations.colors import CEC_BLUE, INPUT_GATE_GREEN, LSTM_T2C
    
    # For mobjects
    circle = Circle(color=CEC_BLUE)
    
    # For LaTeX equations with tex-to-color mapping
    equation = Tex(
        R"s_c(t) = s_c(t-1) + y_{in}(t) \cdot g(net_c(t))",
        t2c=LSTM_T2C
    )
"""

# ============================================================================
# LSTM Component Colors (Hex format for ManimGL)
# ============================================================================

# Cell State (Constant Error Carousel) - Blue
CEC_BLUE = "#3B82F6"

# Input Gate - Green
INPUT_GATE_GREEN = "#22C55E"

# Output Gate - Orange
OUTPUT_GATE_ORANGE = "#F97316"

# Cell Input (g function) - Teal
CELL_INPUT_TEAL = "#14B8A6"

# Gradient/Backpropagation Flow - Red
GRADIENT_RED = "#EF4444"

# Weights and Connections - Gray
WEIGHT_GRAY = "#6B7280"

# Scene Background - Dark
BACKGROUND_DARK = "#1E1E2E"

# Text and Labels - White
TEXT_WHITE = "#FFFFFF"

# Accent and Markers - Yellow
ACCENT_YELLOW = "#FACC15"

# ============================================================================
# LaTeX to Color Mapping (t2c) for Equations
# ============================================================================

LSTM_T2C = {
    # Cell state symbol -> CEC_BLUE
    "s_c": CEC_BLUE,
    "s_{c}": CEC_BLUE,
    
    # Input gate symbol -> INPUT_GATE_GREEN
    "y_{in}": INPUT_GATE_GREEN,
    "y^{in}": INPUT_GATE_GREEN,
    
    # Output gate symbol -> OUTPUT_GATE_ORANGE
    "y_{out}": OUTPUT_GATE_ORANGE,
    "y^{out}": OUTPUT_GATE_ORANGE,
    
    # Cell input function -> CELL_INPUT_TEAL
    "g": CELL_INPUT_TEAL,
    "g(": CELL_INPUT_TEAL,
    
    # Hidden state -> CEC_BLUE (related to cell state)
    "h": CEC_BLUE,
    "h(": CEC_BLUE,
    
    # Generic symbols
    "t": TEXT_WHITE,
    "(": TEXT_WHITE,
    ")": TEXT_WHITE,
}

# ============================================================================
# Utility Functions
# ============================================================================


def hex_to_rgb(hex_color: str) -> tuple:
    """
    Convert a hex color string to an RGB tuple.
    
    Args:
        hex_color (str): Hex color string (e.g., "#3B82F6" or "3B82F6")
    
    Returns:
        tuple: RGB tuple with values in range [0, 1] for ManimGL
    
    Example:
        >>> hex_to_rgb("#3B82F6")
        (0.23137254901960785, 0.5098039215686274, 0.9647058823529412)
    """
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB (0-255 range)
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    
    # Normalize to [0, 1] range for ManimGL
    return (r / 255.0, g / 255.0, b / 255.0)


# ============================================================================
# Color Palettes for Different Visualization Contexts
# ============================================================================

# Gate colors for compact reference
GATE_COLORS = {
    "input": INPUT_GATE_GREEN,
    "output": OUTPUT_GATE_ORANGE,
    "cell": CELL_INPUT_TEAL,
}

# State and flow colors
STATE_COLORS = {
    "cell_state": CEC_BLUE,
    "hidden_state": CEC_BLUE,
    "gradient": GRADIENT_RED,
}

# Structural colors
STRUCTURAL_COLORS = {
    "weights": WEIGHT_GRAY,
    "background": BACKGROUND_DARK,
    "text": TEXT_WHITE,
    "accent": ACCENT_YELLOW,
}
