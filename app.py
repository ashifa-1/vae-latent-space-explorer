import streamlit as st
import torch
import matplotlib.pyplot as plt

from app.utils.model_loader import load_model


LATENT_DIM = 8


@st.cache_resource
def get_model():
    return load_model()


model = get_model()

st.set_page_config(
    page_title="Variational Autoencoder Latent Space Explorer",
    layout="wide"
)

page = st.sidebar.radio(
    "Navigation",
    [
        "Latent Space Explorer",
        "Reconstruction",
        "Training Metrics",
        "KL Analysis"
    ]
)

if page == "Latent Space Explorer":

    st.title("Variational Autoencoder Latent Space Explorer")

    st.write(
        "Adjust latent dimensions and observe how the decoder generates images."
    )

    col1, col2 = st.columns([1, 2])

    with col1:

        latent_values = []

        for i in range(LATENT_DIM):
            value = st.slider(
                f"z{i+1}",
                min_value=-3.0,
                max_value=3.0,
                value=0.0,
                step=0.1
            )
            latent_values.append(value)

    with col2:

        z = torch.tensor(
            [latent_values],
            dtype=torch.float32
        )

        with torch.no_grad():
            generated = model.decoder(z)

        image = generated.squeeze().numpy()

        fig, ax = plt.subplots()

        ax.imshow(
            image,
            cmap="gray"
        )

        ax.axis("off")

        st.pyplot(fig)

elif page == "Reconstruction":

    from app.utils.reconstruction import get_reconstruction

    st.title("Reconstruction")

    original, reconstructed = get_reconstruction(model)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original Image")

        fig1, ax1 = plt.subplots()

        ax1.imshow(
            original,
            cmap="gray"
        )

        ax1.axis("off")

        st.pyplot(fig1)

    with col2:
        st.subheader("Reconstructed Image")

        fig2, ax2 = plt.subplots()

        ax2.imshow(
            reconstructed,
            cmap="gray"
        )

        ax2.axis("off")

        st.pyplot(fig2)

elif page == "Training Metrics":

    import pandas as pd

    st.title("Training Metrics")

    df = pd.read_csv("results/training_log.csv")

    st.subheader("Training Log")
    st.dataframe(df)

    st.subheader("Reconstruction Loss vs Epoch")
    st.line_chart(
        df.set_index("epoch")["reconstruction_loss"]
    )

    st.subheader("KL Divergence vs Epoch")
    st.line_chart(
        df.set_index("epoch")["kl_divergence"]
    )

elif page == "KL Analysis":

    st.title("KL Analysis")

    st.info(
        "KL divergence analysis will be implemented next."
    )