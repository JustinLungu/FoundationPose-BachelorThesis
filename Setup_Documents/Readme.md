# Thesis Project Setup Folder

This folder contains all setup guides and documentation needed to install, configure, and run the full thesis pipeline, which combines **FoundationPose** for 6D pose estimation with **ThreeStudio** for 3D mesh generation. It includes standalone and integrated setup instructions, dataset descriptions, and infrastructure guidelines for both local machines and the Habrok HPC environment.

---

## Contents

| File                                  | Description                                                                                                   |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| `Thesis Project Setup.pdf`            | Unified setup guide for running FoundationPose + ThreeStudio end-to-end                                       |
| `Individual FoundationPose Setup.pdf` | Step-by-step Docker + NVIDIA setup for FoundationPose                                                         |
| `Individual Threestudio Setup.pdf`    | Standalone setup for ThreeStudio (Docker + prompt testing)                                                    |
| `Habrok_Setup.pdf`                    | Guide for running ThreeStudio on Habrok with module setup, CUDA, GPU allocation, and Hugging Face integration |
| `Datasets structure.pdf`              | Visual + textual summary of LINEMOD and Demo dataset folder formats                                           |
| `*.png` images                        | Supporting visuals/screenshots from dataset structure or setup validation                                     |

---

## What You Can Do With This Folder

* Set up FoundationPose individually or in combination with ThreeStudio
* Run ThreeStudio either locally or on the Habrok GPU cluster
* Test integrated inference that auto-generates 3D meshes when missing
* Verify datasets are structured properly before training or evaluation
* Adapt environment variables and `.bashrc` to automate HPC setups

---

## Dataset Support

The folder includes clear documentation for:

* **LINEMOD-style datasets**: multiple frames, RGB, depth, masks, `gt.yml`, intrinsics, and 3D models.
* **Demo-style datasets**: fewer frames, single mask, pose matrices as `.txt` files, and optional texture support.
* Summary of required assets: depth, RGB, masks, mesh, `cam_K.txt`.

---

## Usage Scenarios

| Scenario                                         | Guide                                 |
| ------------------------------------------------ | ------------------------------------- |
| Run FoundationPose with pre-built Docker locally | `Individual FoundationPose Setup.pdf` |
| Generate AI meshes and test prompts              | `Individual Threestudio Setup.pdf`    |
| Run both pipelines on HPC (Habrok)               | `Habrok_Setup.pdf`                    |
| End-to-end pipeline on any system                | `Thesis Project Setup.pdf`            |
