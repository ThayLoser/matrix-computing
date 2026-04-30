import numpy as np

EPSILON = 1e-20

# 1. GAUSS ELIMINATION (khử Gauss - partial pivoting)
def gauss_elimination(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    for k in range(n - 1):
        # 1. Partial pivoting
        max_row = np.argmax(np.abs(A[k:n, k])) + k
        if max_row != k:
            A[[k, max_row]] = A[[max_row, k]]
            b[[k, max_row]] = b[[max_row, k]]
            
        # Kiểm tra pivot quá nhỏ (gần bằng 0)
        if abs(A[k, k]) < EPSILON:
            raise ValueError(f"Khử Gauss thất bại: pivot tại hàng {k} quá gần 0")
            
        # 2. Khử các phần tử
        factors = A[k+1:n, k] / A[k, k]
        A[k+1:n, k:n] -= np.outer(factors, A[k, k:n])
        b[k+1:n] -= factors * b[k]
            
    if abs(A[n-1, n-1]) < EPSILON:
        raise ValueError("Khử Gauss thất bại: Hệ phương trình không có nghiệm duy nhất.")
        
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (b[i] - np.dot(A[i, i+1:], x[i+1:])) / A[i, i]
        
    return x

# 2. LU DECOMPOSITION
def lu_decomposition(A, b):
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    P = np.eye(n, dtype=float)
    L = np.eye(n, dtype=float)
    U = A.copy()
    
    for k in range(n-1):
        p = np.argmax(np.abs(U[k:n, k])) + k
        
        if abs(U[p, k]) < EPSILON:
            continue  # Cột suy biến, bỏ qua

        if p != k:
            U[[k, p], k:n] = U[[p, k], k:n]  
            P[[k, p], :] = P[[p, k], :]      
            if k > 0:
                L[[k, p], 0:k] = L[[p, k], 0:k] # Đổi hàng L (chỉ phần bên trái)

        L[k+1:n, k] = U[k+1:n, k] / U[k, k]
        U[k+1:n, k:n] = U[k+1:n, k:n] - np.outer(L[k+1:n, k], U[k, k:n])
            
    Pb = np.dot(P, b)
    
    # 1. Giải hệ Ly = Pb 
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])
        
    # 2. Giải hệ Ux = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        if abs(U[i, i]) < EPSILON:
            raise ValueError("Hệ phương trình vô nghiệm hoặc có vô số nghiệm.")
        x[i] = (y[i] - np.dot(U[i, i+1:], x[i+1:])) / U[i, i]
        
    return x


# 3. QR (householder)
def qr_householder_solve(A, b):
    R = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    m, n = R.shape
    
    # 1. Khử Householder
    limit = min(m, n)
    for k in range(limit):
        x = R[k:m, k]
        norm_x = np.linalg.norm(x)
        
        if norm_x < EPSILON:
            continue
            
        v = x.copy()
        sign = np.sign(x[0]) if x[0] != 0 else 1.0
        v[0] += sign * norm_x
        v = v / np.linalg.norm(v)
        
        # Cập nhật R và b
        R[k:m, k:n] -= 2.0 * np.outer(v, np.dot(v, R[k:m, k:n]))
        b[k:m] -= 2.0 * v * np.dot(v, b[k:m])
        
    # 2. Thế ngược
    # Nếu hệ có ít phương trình hơn ẩn, bài toán vô số nghiệm, tạm thời chặn lại
    if m < n:
        raise ValueError("Hệ có m < n (ít phương trình hơn ẩn), không có nghiệm duy nhất.")
        
    x_sol = np.zeros(n, dtype=float)
    
    # Dù ma trận ban đầu là m x n (m > n), sau khi phân rã, chỉ cần giải hệ tam giác trên kích thước n x n ở góc trên cùng của R.
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) < EPSILON:
            raise ValueError("Ma trận suy biến hạng (Rank deficient).")
            
        x_sol[i] = (b[i] - np.dot(R[i, i+1:], x_sol[i+1:])) / R[i, i]
        
    return x_sol


# 4. GAUSS SEIDEL
def is_strictly_diagonally_dominant(A):
    # kiểm tra chéo trội ngặt theo hàng
    A = np.array(A, dtype=float)
    n = A.shape[0]

    diag_abs = np.abs(np.diag(A))
    sum_abs = np.sum(np.abs(A), axis=1)

    off_diag_sum = sum_abs - diag_abs
    return np.all(diag_abs > off_diag_sum)

def gauss_seidel(A, b, max_iter=1000, tol=1e-16):
    
    if not is_strictly_diagonally_dominant(A):
        raise ValueError("Ma trận không chéo trội ngặt hàng")
        
    if np.any(np.abs(np.diag(A)) < EPSILON):
        raise ValueError("Lỗi: Có phần tử trên đường chéo bằng 0, không thể chia!")
    
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    n = len(b)
    
    x = np.zeros(n, dtype=float)
    
    for k in range(max_iter):
        x_old = x.copy()
        
        for i in range(n):
            # Tính tổng A[i,j]*x[j] loại trừ phần tử đường chéo
            s = np.dot(A[i, :], x) - A[i, i] * x[i]
            
            x[i] = (b[i] - s) / A[i, i]
            
        error = np.linalg.norm(x - x_old, ord=np.inf)
        
        if error < tol:
            return x
            
    print(f"Cảnh báo: Thuật toán chưa hội tụ sau {max_iter} bước lặp.")
    return x
