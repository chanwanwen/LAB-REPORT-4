import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

PAGE_TITLE = "Sigmoid Activation Function"
st.set_page_config(page_title=PAGE_TITLE, page_icon="🧠", layout="wide")

st.title(PAGE_TITLE)
st.caption("BSD3513 | Introduction to Artificial Intelligence | Lab Report 4")
st.markdown("The Sigmoid function transforms any real-valued input into an output between zero and one.")

st.latex(r"f(x) = \frac{1}{1 + e^{-x}}")

st.sidebar.header("Input Settings")
x_value = st.sidebar.slider("Choose an input value (x)", -10.0, 10.0, 0.0, 0.1)

def sigmoid(x):
    # Clipping avoids numerical overflow for very large magnitudes.
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))

output = float(sigmoid(x_value))

col1, col2 = st.columns(2)
with col1:
    st.metric("Input (x)", f"{x_value:.2f}")
with col2:
    st.metric("Sigmoid Output f(x)", f"{output:.4f}")

x = np.linspace(-10, 10, 400)
y = sigmoid(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, label="Sigmoid function", linewidth=2)
ax.scatter([x_value], [output], s=90, label=f"Selected point ({x_value:.1f}, {output:.3f})", zorder=3)
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.axhline(0.5, linestyle="--", linewidth=1, label="Output = 0.5")
ax.set_title("Sigmoid Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")
if x_value < 0:
    st.write(f"The negative input ({x_value:.2f}) produces an output below 0.5: **{output:.4f}**.")
elif x_value > 0:
    st.write(f"The positive input ({x_value:.2f}) produces an output above 0.5: **{output:.4f}**.")
else:
    st.write("An input of zero produces an output of **0.5**.")

st.subheader("Characteristics")
st.markdown("""
- **Output range:** 0 to 1 (the output approaches, but does not reach, either limit).
- **Advantage:** Produces a smooth output that can be interpreted as a probability in suitable binary-classification output layers.
- **Limitation:** Gradients become very small for strongly positive or negative inputs, which can slow learning in deep networks.
- **Common use:** Output layer for binary classification.
""")
