# Weights

Intel Open Image Denoise (OIDN) 2.x does not ship separate weight files. Unlike
OIDN 1.x, which loaded external `.tza` weight files at runtime, OIDN 2.x
compiles its trained denoising weights directly into the device shared
libraries (e.g. `OpenImageDenoise_device_cpu.dll` on Windows).

Those libraries are installed automatically as part of the `pyoidn` package
(see `requirements.txt`) into your Python environment's
`site-packages/pyoidn/oidn/bin/` folder — there is nothing to download or
place in this folder for the app to work.

This folder is kept as a placeholder for the project's standard structure,
and as the drop-in location if a custom-trained denoising model (e.g. a
PyTorch checkpoint) is ever added alongside or instead of OIDN.
