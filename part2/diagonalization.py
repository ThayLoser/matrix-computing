import numpy as np  # Chỉ dùng để tính trị riêng (eigvals)


# ================================================================
# PHAN 1: CAC HAM TIEN ICH CO BAN
# ================================================================

def copy_matrix(M):
    return [row[:] for row in M]

def mat_mul(A, B):
    n = len(A)
    m = len(A[0])
    p = len(B[0])
    return [[sum(A[i][k] * B[k][j] for k in range(m)) for j in range(p)] for i in range(n)]


# ================================================================
# PHAN 2: KHU GAUSS (REF) VỚI PARTIAL PIVOTING
# ================================================================

def row_echelon_form(M):
    M = copy_matrix(M)
    rows, cols = len(M), len(M[0])
    pivot_cols = []
    row_idx = 0
    row_swaps = 0
    EPS = 1e-9

    for col in range(cols):
        # Partial pivoting
        max_val = EPS
        max_row = None
        for r in range(row_idx, rows):
            if abs(M[r][col]) > max_val:
                max_val = abs(M[r][col])
                max_row = r

        if max_row is None:
            continue

        if max_row != row_idx:
            M[row_idx], M[max_row] = M[max_row], M[row_idx]
            row_swaps += 1

        pivot_cols.append(col)

        for r in range(row_idx + 1, rows):
            if abs(M[r][col]) > EPS:
                factor = M[r][col] / M[row_idx][col]
                for c in range(col, cols):
                    M[r][c] -= factor * M[row_idx][c]
                M[r][col] = 0.0

        row_idx += 1

    return M, pivot_cols, row_swaps


# ================================================================
# PHAN 3: THE NGƯỢC (BACK SUBSTITUTION) TÌM NULL SPACE
# ================================================================

def back_substitution_null_space(ref, pivot_cols, num_cols):
    EPS = 1e-9
    num_rows = len(ref)
    free_cols = [c for c in range(num_cols) if c not in pivot_cols]

    null_vectors = []
    for fc in free_cols:
        vec = [0.0] * num_cols
        vec[fc] = 1.0

        for i in range(len(pivot_cols) - 1, -1, -1):
            pc = pivot_cols[i]
            if i >= num_rows:
                continue
            total = 0.0
            for c in range(pc + 1, num_cols):
                if abs(ref[i][c]) > EPS:
                    total += ref[i][c] * vec[c]
            vec[pc] = -total / ref[i][pc]

        null_vectors.append(vec)

    return null_vectors


# ================================================================
# PHAN 4: TÌM VECTOR RIÊNG
# ================================================================

def build_A_minus_lambdaI(A, lam):
    n = len(A)
    return [
        [A[i][j] - (lam if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]


def find_eigenvectors(A, eigenvalues):
    n = len(A)

    # Lấy trị riêng phân biệt (phần thực)
    unique_eigs = []
    seen = []
    for lam in eigenvalues:
        lam_real = lam.real
        if not any(abs(lam_real - s) < 1e-6 for s in seen):
            unique_eigs.append(lam_real)
            seen.append(lam_real)

    all_eigenvectors = []
    matched_eigenvals = []

    for lam in unique_eigs:
        M = build_A_minus_lambdaI(A, lam)
        ref, pcols, _ = row_echelon_form(M)
        null_vecs = back_substitution_null_space(ref, pcols, n)

        # KHÔNG CHUẨN HÓA: dùng trực tiếp vector thô
        all_eigenvectors.extend(null_vecs)
        matched_eigenvals.extend([lam] * len(null_vecs))

    return all_eigenvectors, matched_eigenvals


# ================================================================
# PHAN 5: KIỂM TRA CHÉO HÓA ĐƯỢC
# ================================================================

def matrix_rank(M):
    """Tính rank bằng số pivot sau khi khu Gauss."""
    _, pivot_cols, _ = row_echelon_form(M)
    return len(pivot_cols)

def is_diagonalizable(A, eigenvectors):
    """Kiểm tra điều kiện chéo hóa: có đúng n vector riêng độc lập tuyến tính."""
    n = len(A)
    if len(eigenvectors) < n:
        return False
    P = build_P(eigenvectors)
    return matrix_rank(P) == n


# ================================================================
# PHAN 6: XÂY DỰNG MA TRẬN P VÀ D
# ================================================================

def build_P(eigenvectors):
    """Ma trận P: mỗi CỘT là một vector riêng."""
    n = len(eigenvectors[0])
    return [[eigenvectors[j][i] for j in range(n)] for i in range(n)]

def build_D(matched_eigenvals):
    """Ma trận D: trị riêng nằm trên đường chéo chính."""
    n = len(matched_eigenvals)
    return [[float(matched_eigenvals[i]) if i == j else 0.0 for j in range(n)] for i in range(n)]


# ================================================================
# PHAN 7: NGHỊCH ĐẢO MA TRẬN P (GAUSS-JORDAN)
# ================================================================

def mat_inverse(M):
    """Tính nghịch đảo bằng Gauss-Jordan."""
    n = len(M)
    EPS = 1e-9

    aug = [M[i][:] + [1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for col in range(n):
        max_row = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[max_row][col]) < EPS:
            raise ValueError(f"Ma trận suy biến tại cột {col}, không có nghịch đảo!")

        aug[col], aug[max_row] = aug[max_row], aug[col]

        pivot = aug[col][col]
        aug[col] = [x / pivot for x in aug[col]]

        for r in range(n):
            if r != col and abs(aug[r][col]) > EPS:
                factor = aug[r][col]
                aug[r] = [aug[r][c] - factor * aug[col][c] for c in range(2 * n)]

    return [aug[i][n:] for i in range(n)]


# ================================================================
# PHAN 8: HÀM CHÍNH — CHEO HÓA MA TRẬN
# ================================================================

def diagonalize(A):
    """
    Cheo hóa ma trận A = P D P⁻¹.

    - Vector riêng KHÔNG được chuẩn hóa (free variable = 1).
    - Chỉ hỗ trợ ma trận vuông thực với trị riêng thực.
    - Trả về (P, D, P_inv) nếu chéo hóa được,
      ngược lại trả về (None, None, None).
    """

    n = len(A)
    # Bước 1: Tính trị riêng
    eigenvalues = np.linalg.eigvals(A)

    # Từ chối nếu có trị riêng phức thực sự
    if any(abs(lam.imag) > 1e-6 for lam in eigenvalues):
        return None, None, None
    
    # Bước 2: Tìm vector riêng (KHÔNG chuẩn hóa)
    eigenvectors, matched_eigenvals = find_eigenvectors(A, eigenvalues)

    # Bước 3: Kiểm tra chéo hóa được
    if not is_diagonalizable(A, eigenvectors):
        return None, None, None

    # Bước 4: Xây dựng P, D, P^-1
    P = build_P(eigenvectors)
    D = build_D(matched_eigenvals)
    P_inv = mat_inverse(P)

    return P, D, P_inv