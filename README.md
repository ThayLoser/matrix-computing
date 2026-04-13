# Ma Trận và Cơ Sở của Tính Toán Khoa Học (Matrices and Fundamentals of Scientific Computing)

**Course:** MTH00051 - Toán Ứng Dụng và Thống Kê (Applied Mathematics and Statistics)

**Institution:** Ho Chi Minh City University of Science (HCMUS)

**Author:**
- Nguyễn Anh Thái
- Nguyễn Đình Tuấn
- Nguyễn Huỳnh Gia Bảo
- Vòng Sau Hậu
- Lương Nhật Tân

## 📌 Project Overview

This repository contains a from-scratch Python implementation of core numerical linear algebra algorithms. The project is divided into three main components:

* **Part 1: Gaussian Elimination & Applications**
    * Implementation of Gaussian elimination with partial pivoting.
    * Solving systems of linear equations via back-substitution.
    * Calculating matrix determinants and inverses (Gauss-Jordan).
    * Finding matrix rank and bases for column, row, and null spaces.
* **Part 2: Matrix Decomposition & Visualization**
    * Implementation of a chosen matrix decomposition (LU, QR, SVD, or Cholesky).
    * Matrix diagonalization (eigenvalues and eigenvectors).
    * Mathematical animation and visualization of the decomposition using Manim.
* **Part 3: Performance Analysis & Numerical Stability**
    * Implementation of the Gauss-Seidel iterative method.
    * Benchmarking runtime and relative error against random matrices of varying sizes ($n \in \{50, 100, 200, 500, 1000\}$).
    * Log-log plotting and stability analysis regarding the matrix Condition Number.

## 🛠️ Technology Stack

* **Language:** Python $\ge 3.10$
* **Visualization:** Manim Community v0.18.0
* **Data Analysis & Plotting:** Matplotlib, Jupyter Notebook
* **Verification (Not used for core logic):** NumPy, SciPy, SymPy

## 📂 Repository Structure

The codebase strictly follows the required project structure:

```text
Group_<ID>/
|-- README.md
|-- requirements.txt
|-- report/
|   |-- report.pdf
|   |-- report.tex
|-- part1/
|   |-- gaussian.py
|   |-- determinant.py
|   |-- inverse.py
|   |-- rank_basis.py
|   |-- part1_demo.ipynb
|-- part2/
|   |-- decomposition.py
|   |-- diagonalization.py
|   |-- manim_scene.py
|   |-- demo_video.mp4
|-- part3/
|   |-- solvers.py
|   |-- benchmark.py
|   |-- analysis.ipynb
```

## 🚀 Setup & Installation

1. Clone this repository.
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. To run the Manim visualization for Part 2:
   ```bash
   manim -pqh part2/manim_scene.py <SceneClassName>
   ```
4. Explore the Jupyter notebooks (`part1_demo.ipynb` and `analysis.ipynb`) to see the implementations and benchmarks in action.
