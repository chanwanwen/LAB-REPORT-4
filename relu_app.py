import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

PAGE_TITLE = "ReLU Activation Function"
st.set_page_config(page_title=PAGE_TITLE, page_icon="🧠", layout="wide")

st.title(PAGE_TITLE)
st.caption("BSD3513 | Introduction to Artificial Intelligence | Lab Report 4")
st.markdown("The Rectified Linear Unit (ReLU) returns zero for negative inputs and returns the input itself for positive inputs.")

st.latex(r"f(x) = \max(0, x)")

st.sidebar.header("Input Settings")
x_value = st.sidebar.slider("Choose an input value (x)", -10.0, 10.0, 2.0, 0.1)

output = max(0.0, x_value)

col1, col2 = st.columns(2)
with col1:
    st.metric("Input (x)", f"{x_value:.2f}")
with col2:
    st.metric("ReLU Output f(x)", f"{output:.2f}")

x = np.linspace(-10, 10, 400)
y = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(x, y, label="ReLU: f(x) = max(0, x)", linewidth=2)
ax.scatter([x_value], [output], s=90, label=f"Selected point ({x_value:.1f}, {output:.1f})", zorder=3)
ax.axhline(0, linewidth=0.8)
ax.axvline(0, linewidth=0.8)
ax.set_title("ReLU Activation Function")
ax.set_xlabel("Input (x)")
ax.set_ylabel("Output f(x)")
ax.grid(True, alpha=0.3)
ax.legend()
st.pyplot(fig)
plt.close(fig)

st.subheader("Interpretation")
if x_value < 0:
    st.write(f"Because the input is negative ({x_value:.2f}), ReLU returns **0**. Negative values are not passed through.")
elif x_value > 0:
    st.write(f"Because the input is positive ({x_value:.2f}), ReLU returns the same value: **{output:.2f}**.")
else:
    st.write("When the input is zero, the ReLU output is also **0**.")

st.subheader("Characteristics")
st.markdown("""
- **Output range:** 0 to positive infinity.
- **Advantage:** Simple and computationally efficient; helps reduce the vanishing-gradient problem for positive inputs.
- **Limitation:** Neurons can become inactive when they repeatedly receive negative inputs (the dying ReLU problem).
- **Common use:** Hidden layers in many feedforward and convolutional neural networks.
""")
