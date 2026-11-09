import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

PAGE_TITLE = "Tanh Activation Function"
st.set_page_config(page_title=PAGE_TITLE, page_icon="🧠", layout="wide")

st.title(PAGE_TITLE)
st.caption("BSD3513 | Introduction to Artificial Intelligence | Lab Report 4")
st.markdown("The hyperbolic tangent (Tanh) function maps real-valued inputs to outputs between negative one and positive one.")

st.latex(r"f(x) = \tanh(x)")

st.sidebar.header("Input Settings")
x_value = st.sidebar.slider("Choose an input value (x)", -10.0, 10.0, 0.0, 0.1)

output = float(np.tanh(x_value))

col1, col2 = st.columns(2)
with col1:
    st.metric("Input (x)", f"{x_value:.2f}")
with col2:
    st.metric("Tanh Output f(x)", f"{output:.4f}")

x = np.linspace(-10, 10, 400)
y = np.tanh(x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, label="Tanh function", linewidth=2)
ax.scatter([x_value], [output], s=90, label=f"Selected point ({x_value:.1f}, {output:.3f})", zorder=3)
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.axhline(1, linestyle="--", linewidth=1, alpha=0.7)
ax.axhline(-1, linestyle="--", linewidth=1, alpha=0.7)
ax.set_title("Tanh Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.set_ylim(-1.1, 1.1)
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")
if x_value < 0:
    st.write(f"The negative input ({x_value:.2f}) produces a negative output: **{output:.4f}**.")
elif x_value > 0:
    st.write(f"The positive input ({x_value:.2f}) produces a positive output: **{output:.4f}**.")
else:
    st.write("An input of zero produces an output of **0**.")

st.subheader("Characteristics")
st.markdown("""
- **Output range:** -1 to 1 (the output approaches, but does not reach, either limit).
- **Advantage:** Zero-centred output, which can make optimisation easier than with Sigmoid in some settings.
- **Limitation:** Like Sigmoid, Tanh can suffer from vanishing gradients for inputs with large absolute values.
- **Common use:** Hidden layers in some neural networks, especially when zero-centred activations are useful.
""")
