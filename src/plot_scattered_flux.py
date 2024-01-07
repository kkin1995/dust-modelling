import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv

load_dotenv()
DATA = os.environ.get("DATA")

flux_data = pd.read_csv(os.path.join(DATA, "processed", "flux_data.csv"))
stars = flux_data.loc[:, "Star"].values
# star = stars[0]
star = "88469"

flux = np.load(
    os.path.join(DATA, "processed", "single_scatter_model", f"{star}.npy")
)  # (3, 100, 10, 10)

PLOT_FLUX_VS_ALBEDO = False
PLOT_FLUX_VS_PHASE = False
PLOT_FLUX_VS_ANGLE = False
PLOT_FLUX_VS_ANGLE_FOR_ALBEDO = True
PLOT_FLUX_VS_ANGLE_FOR_PHASE = False

wavelength = [1100, 1500, 2300]  # Angstroms
angle = np.arange(0, 100, 1) * 0.1  # [0, 1, 2, ..., 10] Degrees
albedo = np.arange(0, 1.0, 0.1)
phase = np.arange(0, 1.0, 0.1)

# 1100 Angstroms

if PLOT_FLUX_VS_PHASE:
    iwave = 0  # wavelength is 1100 Angstroms
    iangle = 60
    fig, ax = plt.subplots()
    for ialbedo in range(10):
        flux_with_phase = flux[iwave, iangle, ialbedo, :]
        ax.plot(phase, flux_with_phase, label="a = " + str(round(albedo[ialbedo], 2)))

    ax.legend()
    ax.set_title(
        f"HIP {star} | Angle: {angle[iangle]} | Wavelength: {wavelength[iwave]}"
    )
    ax.set_xlabel("Phase g")
    ax.set_ylabel("Flux")
    plt.show()

if PLOT_FLUX_VS_ALBEDO:
    iwave = 0  # wavelength is 1100 Angstroms
    iangle = 20
    fig, ax = plt.subplots()
    for iphase in range(10):
        flux_with_albedo = flux[iwave, iangle, :, iphase]
        ax.scatter(
            albedo, flux_with_albedo, label="g = " + str(round(phase[iphase], 2)), s=10
        )

    ax.legend()
    ax.set_title(
        f"HIP {star} | Angle: {angle[iangle]} | Wavelength: {wavelength[iwave]}"
    )
    ax.set_xlabel("Albedo")
    ax.set_ylabel("Flux")
    plt.show()

# 1100 Angstroms
if PLOT_FLUX_VS_ANGLE:
    iwave = 0  # wavelength is 1100 Angstroms
    ialbedo = 5  # a is 0.1
    iphase = 5  # g is 0.1
    fig, ax = plt.subplots()
    flux_with_angle = flux[iwave, :, ialbedo, iphase]
    plt.title(
        f"HIP {star} | a = {albedo[ialbedo]} | g = {phase[iphase]} | Wavelength: {wavelength[iwave]}"
    )
    plt.xlabel("Angle")
    plt.ylabel("Flux")
    plt.plot(angle, flux_with_angle)
    plt.show()

if PLOT_FLUX_VS_ANGLE_FOR_ALBEDO:
    iwave = 0  # wavelength is 1100 Angstroms
    iphase = 9  # g is 0.9
    fig, ax = plt.subplots()
    for ialbedo in range(10):
        flux_with_angle = flux[iwave, :, ialbedo, iphase]
        ax.plot(
            angle[0:5],
            flux_with_angle[0:5],
            label="a = " + str(round(albedo[ialbedo], 2)),
        )

    ax.legend()
    ax.set_title(f"HIP {star} | g = {phase[iphase]} | Wavelength: {wavelength[iwave]}")
    ax.set_xlabel("Angle")
    ax.set_ylabel("Flux")
    plt.show()

if PLOT_FLUX_VS_ANGLE_FOR_PHASE:
    iwave = 0  # wavelength is 1100 Angstroms
    ialbedo = 9  # a is 0.9
    fig, ax = plt.subplots()
    for iphase in range(10):
        flux_with_angle = flux[iwave, :, ialbedo, iphase]
        ax.plot(
            angle[0:5],
            flux_with_angle[0:5],
            label="g = " + str(round(phase[iphase], 2)),
        )

    ax.legend()
    ax.set_title(
        f"HIP {star} | a = {albedo[ialbedo]} | Wavelength: {wavelength[iwave]}"
    )
    ax.set_xlabel("Angle")
    ax.set_ylabel("Flux")
    plt.show()
