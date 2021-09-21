# Depth from Focus
**Depth from Focus** or **Shape from Focus** is a method of reconstructing a **3D Object Depth Map** and it's corresponding **All-Focused image**
using focus as a que for depth. This is efficient and isn't as costly as a _stereo camera_.

>According to the physics of lenses, blur circles are formed when an object is not located at the working distance (plane of focus).
If we see an image that is perfectly in focus throughout, then it is saying that the object is in a frontoparallel planar world. If only
some parts are in focus then it is hinting that the object has a 3D structure!

To make use of this effectively, we keep the **Depth of Field** as small as possible and take a stack of images by changing the focus each time.
This is then processesed pixel-by-pixel by a **focus operator** to find the depth at which it is most likely in focus.

## Examples
| First Image in Focus Stack | Last Image in Focus Stack | Reconstructed All-Focus Image |
| :------------------------: | :-----------------------: | :---------------------------: |
| <img src='https://user-images.githubusercontent.com/64144419/134203688-da052c57-f77b-443e-9d92-fa4c0a2cd7ed.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134203823-f910fb53-b5b0-460c-aee6-c6430633b5c9.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134203906-3d886fcc-820f-4940-b9bd-0b1b73c63420.png' width=300> |

| 2D Depth Plot | 3D Depth Plot |
| :-----------: | :-----------: |
| <img src='https://user-images.githubusercontent.com/64144419/134204165-6ab94953-e1c3-4283-867a-cfa304f754b1.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134204378-49707fb2-1180-4b9e-bed9-ff54a1135c0f.png' width=300> |

| First Image in Focus Stack | Last Image in Focus Stack | Reconstructed All-Focus Image |
| :------------------------: | :-----------------------: | :---------------------------: |
| <img src='https://user-images.githubusercontent.com/64144419/134191705-8b034206-86bf-495f-b525-7da886ffe254.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134191868-c24474fd-88be-42a9-acff-e42500347bc6.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134191955-2b8abaa8-218d-4b27-b5f8-dbd400e302d1.png' width=300> |

| 2D Depth Plot | 3D Depth Plot |
| :-----------: | :-----------: |
| <img src='https://user-images.githubusercontent.com/64144419/134192148-71f33673-3c15-4eff-9048-04793a77889a.png' width=300> | <img src='https://user-images.githubusercontent.com/64144419/134192571-21a68f45-0364-451b-8994-3c780ccdc69e.png' width=300> |

### Assumptions:
* This project assumes that the images do not undergo axial parallax (or is corrected) when changing focus. This is always true if we use a **telecentric lens** else it has to be preprocessed for alignment.

### Implementation:
* Apply Focus Operator, **Sum Modified Laplacian** (SML) with kernel size 3.
* **Gaussian Interpolation** to find the best depth, to counter discretization.
* Use the generated depth map to reconstruct an **All-Focused image**.
* Apply Median Filter to cleanup some noise.
* Generate a **2D Grayscaled Depth Plot**.
* Generate a **3D Depth Surface Plot**.
* Use `python3 dff.py` to run the script.

### Dataset:
The dataset folder contains a compilation of various aligned focus stack I could find online and on GitHub.
I hold no copyright for any of the images in the dataset. This is for learning and education purposes only.

## Further Explorations:
* Use an Optical Flow / SIFT algorithms to correct axial parallax.
* Use Graph Cut and segmentation approaches to fine-tune the depth map.
