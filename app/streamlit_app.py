"""Image AI Lab — Streamlit frontend for the OIDN / VAE / GAN modules.

Run with: streamlit run app/streamlit_app.py
"""

import io
import sys
from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

VAE_CHECKPOINT = ROOT_DIR / "models" / "vae" / "vae.pt"
GAN_CHECKPOINTS = {
    "MLP (basic)": ROOT_DIR / "models" / "gan" / "generator.pt",
    "DCGAN (conv)": ROOT_DIR / "models" / "gan" / "generator_dcgan.pt",
}

st.set_page_config(page_title="Image AI Lab", layout="wide")
st.title("Image AI Lab")
st.caption("Explore denoising, reconstruction, and image generation.")

tab_oidn, tab_vae, tab_gan = st.tabs(["OIDN", "VAE", "GAN"])


def read_uploaded_image(uploaded_file) -> Image.Image | None:
    """Try to open an uploaded file as an image, showing a friendly error on failure."""
    try:
        return Image.open(uploaded_file)
    except UnidentifiedImageError:
        st.error("That file doesn't look like a valid image. Try a PNG or JPG.")
        return None
    except Exception:
        st.error("Couldn't read that file. Try a different image.")
        return None


def image_download_button(image: Image.Image, label: str, file_name: str):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    st.download_button(label, data=buffer.getvalue(), file_name=file_name, mime="image/png")


# ---------------------------------------------------------------------------
# OIDN tab
# ---------------------------------------------------------------------------
with tab_oidn:
    st.subheader("OIDN denoising")
    st.write(
        "Runs Intel Open Image Denoise on an image. OIDN only removes noise — "
        "it doesn't generate or reconstruct anything, it just cleans up what's there."
    )

    try:
        from oidn.inference import denoise, add_synthetic_noise, SAMPLE_NOISY
        oidn_import_error = None
    except Exception as exc:
        oidn_import_error = exc

    if oidn_import_error is not None:
        st.error(
            "OIDN isn't available in this environment. "
            f"Install it with `pip install pyoidn` and restart the app. ({oidn_import_error})"
        )
    else:
        col_input, col_output = st.columns(2)

        with col_input:
            uploaded = st.file_uploader(
                "Upload a noisy image", type=["png", "jpg", "jpeg", "bmp"], key="oidn_upload"
            )
            use_sample = st.checkbox(
                "Use bundled sample image instead", value=not uploaded, key="oidn_sample"
            )
            add_noise = st.checkbox("Add synthetic noise before denoising", key="oidn_add_noise")
            noise_sigma = st.slider(
                "Noise strength", 5, 60, 25, disabled=not add_noise, key="oidn_sigma"
            )
            run = st.button("Denoise", type="primary")

        source_image = None
        if uploaded is not None and not use_sample:
            source_image = read_uploaded_image(uploaded)
        elif SAMPLE_NOISY.exists():
            source_image = Image.open(SAMPLE_NOISY)

        if source_image is not None:
            noisy_image = (
                add_synthetic_noise(source_image, noise_sigma) if add_noise else source_image
            )
            with col_input:
                st.image(noisy_image, caption="Noisy image", width="stretch")

            with col_output:
                if run:
                    with st.spinner("Running OIDN..."):
                        try:
                            denoised = denoise(noisy_image)
                        except Exception as exc:
                            st.error(f"Denoising failed: {exc}")
                            denoised = None
                    if denoised is not None:
                        st.image(denoised, caption="After denoising", width="stretch")
                        image_download_button(denoised, "Download result", "denoised.png")
                else:
                    st.info("Click **Denoise** to see the result here.")


# ---------------------------------------------------------------------------
# VAE tab
# ---------------------------------------------------------------------------
with tab_vae:
    st.subheader("VAE reconstruction & generation")
    st.write(
        "Trained on MNIST digits only — reconstructions will look best on simple "
        "grayscale digit-like images, not arbitrary photos."
    )

    if not VAE_CHECKPOINT.exists():
        st.warning(
            "No trained VAE checkpoint found. Train one first with:\n\n"
            "```\npython vae/train.py\n```"
        )
    else:
        try:
            from vae.inference import load_model as load_vae_model, reconstruct, generate as vae_generate
            vae_model = load_vae_model(VAE_CHECKPOINT)
            vae_load_error = None
        except Exception as exc:
            vae_model = None
            vae_load_error = exc

        if vae_load_error is not None:
            st.error(f"Couldn't load the VAE checkpoint: {vae_load_error}")
        else:
            vae_recon_tab, vae_gen_tab = st.tabs(["Reconstruct", "Generate"])

            with vae_recon_tab:
                col_in, col_out = st.columns(2)
                with col_in:
                    vae_upload = st.file_uploader(
                        "Upload an image to reconstruct", type=["png", "jpg", "jpeg", "bmp"],
                        key="vae_upload",
                    )
                if vae_upload is not None:
                    vae_source = read_uploaded_image(vae_upload)
                    if vae_source is not None:
                        with col_in:
                            st.image(vae_source, caption="Original image", width="stretch")
                        with col_out:
                            try:
                                recon = reconstruct(vae_model, vae_source)
                                st.image(recon, caption="VAE reconstruction", width="stretch")
                            except Exception as exc:
                                st.error(f"Reconstruction failed: {exc}")
                else:
                    st.info("Upload an image to see its VAE reconstruction.")

            with vae_gen_tab:
                n_samples = st.slider("Number of samples", 1, 16, 8, key="vae_n")
                if st.button("Generate random samples", key="vae_gen_button"):
                    try:
                        samples = vae_generate(vae_model, n_samples)
                        st.image(samples, caption=["Random samples from the latent space"] * len(samples))
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")


# ---------------------------------------------------------------------------
# GAN tab
# ---------------------------------------------------------------------------
with tab_gan:
    st.subheader("GAN generation")
    st.write("Generates new images from random noise. There's no input image here — that's the point of a GAN.")

    architecture = st.radio(
        "Architecture",
        list(GAN_CHECKPOINTS.keys()),
        horizontal=True,
        key="gan_architecture",
        help="MLP is the original basic GAN. DCGAN (conv-based) gives smoother, less speckled samples and less mode collapse — see gan/README.md for the comparison.",
    )
    gan_checkpoint = GAN_CHECKPOINTS[architecture]

    if not gan_checkpoint.exists():
        train_cmd = "python gan/train.py" if architecture == "MLP (basic)" else "python gan/train.py --architecture dcgan"
        st.warning(
            f"No trained checkpoint found for {architecture}. Train one first with:\n\n"
            f"```\n{train_cmd}\n```"
        )
    else:
        try:
            from gan.inference import load_generator, generate as gan_generate
            gan_model = load_generator(gan_checkpoint)
            gan_load_error = None
        except Exception as exc:
            gan_model = None
            gan_load_error = exc

        if gan_load_error is not None:
            st.error(f"Couldn't load the GAN checkpoint: {gan_load_error}")
        else:
            n_images = st.slider("Number of images", 1, 16, 8, key="gan_n")
            if st.button("Generate", type="primary", key="gan_gen_button"):
                with st.spinner("Generating..."):
                    try:
                        samples = gan_generate(gan_model, n_images)
                        st.image(samples, caption=[f"{architecture} sample"] * len(samples))
                    except Exception as exc:
                        st.error(f"Generation failed: {exc}")
